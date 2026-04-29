# -*- coding: utf-8 -*-
"""JSONL/CSV/POSCAR/checkpoint/progress 持久化。

主进程在这里统一写结果：trial evaluation 先变成 TrialRecord，再写 JSONL/CSV，必要时写 POSCAR，
最后推进 progress_state 和 checkpoint_state。worker 不调用本模块的写盘函数。
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import PROGRAM_TAG, PROGRAM_VERSION, RECORD_SCHEMA, REQUIRED_REDUCTION_BACKEND
from .debug_tools import debug_stage_event
from .geometry import lattice_volume
from .io_poscar import write_poscar
from .models import (
    Crystal,
    DebugOptions,
    DerivedScanConfig,
    PersistResult,
    ProgressState,
    ReducedStructureData,
    RunPaths,
    RunRuntimeState,
    RunSummary,
    ScanConfig,
    ScanPlan,
    SingleTrialTask,
    TrialEvaluation,
    TrialRecord,
    TrialTask,
)
from .path_utils import _safe_filename
from .structure_reduction import reduce_distorted_structure

def _record_schema() -> List[str]:
    return list(RECORD_SCHEMA)

def make_base_record() -> Dict[str, Any]:
    return {k: None for k in RECORD_SCHEMA}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

def build_trial_record(task: TrialTask, evaluation: TrialEvaluation, persist_result: PersistResult) -> TrialRecord:
    if task.trial_id is None:
        raise RuntimeError("Trial task is missing trial_id during record build.")
    return TrialRecord(
        trial=int(task.trial_id),
        trial_type=evaluation.kind,
        mode_indices=evaluation.mode_indices,
        op_direction=evaluation.op_direction,
        amps=evaluation.amps,
        sa=evaluation.sa,
        sb=evaluation.sb,
        sc=evaluation.sc,
        sg=evaluation.sg,
        sg_sym=evaluation.sg_sym,
        pg=evaluation.pg,
        hit=evaluation.hit,
        label=evaluation.label,
        poscar=persist_result.poscar_path,
        combo_target_sgs=evaluation.combo_target_sgs,
    )

def print_phase_progress(task: TrialTask, evaluation: TrialEvaluation, config: ScanConfig) -> None:
    if task.trial_id is None:
        return
    want_print = ((task.trial_id % int(config.print_every_n_trials)) == 0) or (bool(config.print_also_on_hit) and bool(evaluation.hit))
    if not want_print:
        return

    if isinstance(task, SingleTrialTask):
        print(
            f"[PROGRESS][single] trial={task.trial_id} "
            f"block={task.mode_block_i+1}/{task.n_mode_blocks} mode={task.mode_i+1}/{task.n_modes} "
            f"amp_i={task.amp_i+1}/{task.n_amps} strain_i={task.strain_i+1}/{task.n_strains} "
            f"sg={evaluation.sg} hit={evaluation.hit}"
        )
    else:
        print(
            f"[PROGRESS][combo]  trial={task.trial_id} "
            f"spec={task.spec_i+1}/{task.n_specs} dir={task.dir_i+1}/{task.n_dirs} "
            f"rho={task.rho_i+1}/{task.n_rhos} strain={task.cstrain_i+1}/{task.n_cstrains} "
            f"sg={evaluation.sg} hit={evaluation.hit}"
        )

class ProgressTracker:
    """checkpoint_state 的持久化管理器。

    checkpoint.json 记录 phase 游标、已写记录数、最后 trial id 和每个 SG 的结构写出计数；
    schema.json 固定 CSV 字段，保证断点续跑时字段顺序兼容。
    """

    def __init__(self, out_dir: Path, every: int = 10000):
        self.out_dir = Path(out_dir)
        self.every = int(every)
        self.ckpt_path = self.out_dir / "checkpoint.json"
        self.schema_path = self.out_dir / "schema.json"
        self.state = ProgressState()

    def load_if_exists(self) -> Optional[ProgressState]:
        if self.ckpt_path.exists():
            d = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
            if "write_counts" not in d:
                d["write_counts"] = {}
            if "sg_counter" not in d:
                d["sg_counter"] = {}
            valid_fields = {field.name for field in fields(ProgressState)}
            self.state = ProgressState(**{k: v for k, v in d.items() if k in valid_fields})
            return self.state
        return None

    def should_checkpoint(self) -> bool:
        return self.state.written > 0 and (self.state.written % self.every == 0)

    def save(self) -> None:
        tmp = self.ckpt_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.ckpt_path)

    def save_schema_if_missing(self, fieldnames: List[str]) -> None:
        if self.schema_path.exists():
            return
        tmp = self.schema_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"fieldnames": fieldnames}, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.schema_path)

    def load_schema_if_exists(self) -> Optional[List[str]]:
        if self.schema_path.exists():
            d = json.loads(self.schema_path.read_text(encoding="utf-8"))
            fieldnames = d.get("fieldnames", None)
            if isinstance(fieldnames, list) and fieldnames:
                return [str(x) for x in fieldnames]
        return None


class ResultWriter:
    """
    流式 writer_state，避免长扫描把结果全部留在内存中。

    JSONL/CSV schema 保持历史兼容；每 N 条记录 flush+fsync，并支持已有文件上的 append 恢复。
    """

    def __init__(self, out_dir: Path, tracker: ProgressTracker, write_csv: bool = True, flush_every: int = 10000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.flush_every = int(flush_every)
        self.tracker = tracker

        self.jsonl_path = self.out_dir / "scan_results.jsonl"
        self.csv_path = self.out_dir / "scan_results.csv"

        resume = self.jsonl_path.exists() and (self.csv_path.exists() or not write_csv)

        self._f_jsonl = self.jsonl_path.open("a" if resume else "w", encoding="utf-8", buffering=1)

        self._write_csv = bool(write_csv)
        self._f_csv = None
        self._csv_writer = None
        self._csv_fieldnames = None

        if self._write_csv:
            self._f_csv = self.csv_path.open("a" if resume else "w", newline="", encoding="utf-8")

        schema = self.tracker.load_schema_if_exists()
        if schema:
            self._csv_fieldnames = schema
            if self._write_csv:
                if self._f_csv is None:
                    raise RuntimeError("CSV writer requested but CSV file handle is not open.")
                self._csv_writer = csv.DictWriter(self._f_csv, fieldnames=self._csv_fieldnames, extrasaction="ignore")
                if self.csv_path.stat().st_size == 0:
                    self._csv_writer.writeheader()

        self.n_written = 0

    def _init_csv(self, rec: Dict[str, Any]) -> None:
        if not self._write_csv:
            return
        if self._csv_writer is not None:
            return
        self._csv_fieldnames = list(rec.keys())
        self.tracker.save_schema_if_missing(self._csv_fieldnames)
        if self._f_csv is None:
            raise RuntimeError("CSV writer requested but CSV file handle is not open.")
        self._csv_writer = csv.DictWriter(self._f_csv, fieldnames=self._csv_fieldnames, extrasaction="ignore")
        if self.csv_path.stat().st_size == 0:
            self._csv_writer.writeheader()

    def write(self, rec: Dict[str, Any]) -> None:
        self._f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self._write_csv:
            if self._csv_writer is None:
                self._init_csv(rec)
            if self._csv_writer is None:
                raise RuntimeError("CSV writer requested but was not initialized.")
            self._csv_writer.writerow(rec)

        self.n_written += 1
        if self.n_written % self.flush_every == 0:
            self.flush_fsync()

    def flush_fsync(self) -> None:
        self._f_jsonl.flush()
        os.fsync(self._f_jsonl.fileno())
        if self._write_csv and self._f_csv is not None:
            self._f_csv.flush()
            os.fsync(self._f_csv.fileno())

    def close(self) -> None:
        try:
            self.flush_fsync()
        except Exception:
            pass
        try:
            self._f_jsonl.close()
        finally:
            if self._f_csv is not None:
                self._f_csv.close()


def maybe_write_trial_structure(
    trial: Crystal,
    out_dir: Path,
    trial_id: int,
    kind: str,
    label: str,
    sg: int,
    sg_sym: str,
    pg: Optional[str],
    hit: bool,
    write_counts: Dict[int, int],
    derived: DerivedScanConfig,
    config: ScanConfig,
    evaluation: TrialEvaluation,
) -> PersistResult:
    want_write = (
        bool(derived.write_all_trial_structures)
        or (bool(derived.write_hit_structures) and bool(hit))
        or bool(derived.write_per_sg_structures)
    )
    if not want_write:
        return PersistResult(poscar_path=None)

    if derived.write_limit_per_sg is not None:
        if write_counts.get(int(sg), 0) >= int(derived.write_limit_per_sg):
            return PersistResult(poscar_path=None)

    subdir = "structures_by_sg" if (derived.write_all_trial_structures or derived.write_per_sg_structures) else "hit_structures_by_sg"
    folder = Path(out_dir) / subdir / f"sg{int(sg):03d}"
    fname = _safe_filename(f"trial{int(trial_id):08d}_{kind}_{label}_sg{int(sg)}")
    path = folder / f"{fname}.vasp"

    comment = (
        f"{PROGRAM_TAG} v{PROGRAM_VERSION} | "
        f"trial={trial_id} | kind={kind} | SG={sg} ({sg_sym}) | PG={pg} | {label}"
    )
    output_crystal = trial
    reduction_data: Optional[ReducedStructureData] = None
    output_cell_kind = "unreduced_supercell"
    unreduced_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    if bool(config.reduce_distorted_output_cell):
        if config.reduce_distorted_max_atoms is not None and trial.nsites <= int(config.reduce_distorted_max_atoms):
            reduction_data = ReducedStructureData(
                reduction_attempted=False,
                reduction_successful=False,
                backend=REQUIRED_REDUCTION_BACKEND,
                original_nsites=int(trial.nsites),
                reduced_nsites=int(trial.nsites),
                original_volume=float(lattice_volume(trial.lattice)),
                reduced_volume=float(lattice_volume(trial.lattice)),
                volume_ratio=1.0,
                original_lattice=np.asarray(trial.lattice, dtype=float).copy(),
                reduced_lattice=None,
                original_frac=np.asarray(trial.frac, dtype=float).copy(),
                reduced_frac=None,
                reduced_numbers=None,
                reduced_symbols=None,
                symprec=float(config.reduce_distorted_symprec),
                angle_tolerance=float(config.reduce_distorted_angle_tolerance),
                max_reconstruction_error=None,
                formula_preserved=True,
                reason=(
                    "Skipped reduction because original atom count does not exceed "
                    f"reduce_distorted_max_atoms={config.reduce_distorted_max_atoms}."
                ),
                diagnostics={"skipped_due_to_atom_count_threshold": True},
            )
        else:
            output_crystal, reduction_data = reduce_distorted_structure(
                trial,
                evaluation.displacement,
                backend=REQUIRED_REDUCTION_BACKEND,
                symprec=float(config.reduce_distorted_symprec),
                angle_tolerance=float(config.reduce_distorted_angle_tolerance),
                verify=bool(config.reduce_distorted_verify),
                verify_tol=float(config.reduce_distorted_verify_tol),
                min_volume_ratio=float(config.reduce_distorted_min_volume_ratio),
                skip_if_amplitude_below=float(config.reduce_distorted_skip_if_amplitude_below),
                strict=bool(config.reduce_distorted_strict),
            )
        if reduction_data is not None and reduction_data.reduction_successful:
            output_cell_kind = "reduced_distorted_primitive"
            if bool(config.reduce_distorted_keep_unreduced_copy):
                unreduced_path = path.with_name(f"{path.stem}.unreduced{path.suffix}")
                write_poscar(trial, unreduced_path, comment=comment + " | unreduced distorted supercell")
        if reduction_data is not None and bool(config.reduce_distorted_log_mapping):
            print(
                "[REDUCE] "
                f"attempted={reduction_data.reduction_attempted} "
                f"backend={reduction_data.backend} "
                f"original_nsites={reduction_data.original_nsites} "
                f"reduced_nsites={reduction_data.reduced_nsites} "
                f"volume_ratio={reduction_data.volume_ratio:.3f} "
                f"max_reconstruction_error={reduction_data.max_reconstruction_error} "
                f"success={reduction_data.reduction_successful} "
                f"reason={reduction_data.reason}"
            )

    write_poscar(output_crystal, path, comment=comment)

    if reduction_data is not None:
        metadata_path = path.with_name(f"{path.stem}.structure_metadata.json")
        payload = {
            "output_cell_kind": output_cell_kind,
            "final_poscar": "reduced" if output_cell_kind == "reduced_distorted_primitive" else "unreduced",
            "unreduced_poscar_saved": unreduced_path is not None,
            "unreduced_poscar": None if unreduced_path is None else str(unreduced_path),
            "source_modes": evaluation.mode_indices,
            "source_supercell_matrix": None
            if evaluation.source_supercell_matrix is None
            else np.asarray(evaluation.source_supercell_matrix, dtype=int).tolist(),
            "source_mode_cell_kind": evaluation.source_mode_cell_kind,
            "reduction": _json_safe(asdict(reduction_data)),
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    write_counts[int(sg)] = write_counts.get(int(sg), 0) + 1

    return PersistResult(
        poscar_path=str(path),
        reduced_structure=reduction_data,
        output_cell_kind=output_cell_kind,
        unreduced_poscar_path=None if unreduced_path is None else str(unreduced_path),
        metadata_path=None if metadata_path is None else str(metadata_path),
    )


def persist_trial_artifacts(
    task: TrialTask,
    evaluation: TrialEvaluation,
    run_dir: Path,
    write_counts: Dict[int, int],
    derived: DerivedScanConfig,
    result_write_enabled: bool,
    config: ScanConfig,
) -> PersistResult:
    """根据 SAVE_POLICY 写出 POSCAR，并返回可进入 record 的路径信息。"""
    if task.trial_id is None:
        raise RuntimeError("Trial task is missing trial_id during persistence.")
    poscar_path = None
    if result_write_enabled:
        return maybe_write_trial_structure(
            trial=evaluation.trial_crystal,
            out_dir=run_dir,
            trial_id=int(task.trial_id),
            kind=str(evaluation.kind),
            label=str(evaluation.label),
            sg=int(evaluation.sg),
            sg_sym=str(evaluation.sg_sym),
            pg=evaluation.pg,
            hit=bool(evaluation.hit),
            write_counts=write_counts,
            derived=derived,
            config=config,
            evaluation=evaluation,
        )
    return PersistResult(poscar_path=poscar_path)


def write_trial_record(record: TrialRecord, writer: ResultWriter) -> None:
    """把结构化 TrialRecord 写入 JSONL/CSV；字段由 ResultWriter/RECORD_SCHEMA 控制。"""
    writer.write(record.to_dict())


def sync_tracker_state(runtime: RunRuntimeState) -> None:
    runtime.tracker.state.written = int(runtime.writer.n_written)
    runtime.tracker.state.last_trial_id = int(runtime.trial_id)
    runtime.tracker.state.write_counts = {str(k): int(v) for k, v in runtime.write_counts.items()}
    runtime.tracker.state.sg_counter = {str(k): int(v) for k, v in runtime.sg_counter.items()}


def update_progress_cursor(progress_state: ProgressState, task: TrialTask) -> None:
    if isinstance(task, SingleTrialTask):
        progress_state.phase = "single"
        progress_state.mode_block_i = int(task.mode_block_i)
        progress_state.mode_i = int(task.mode_i)
        progress_state.amp_i = int(task.amp_i)
        progress_state.strain_i = int(task.strain_i) + 1
        return

    progress_state.phase = "combo"
    progress_state.spec_i = int(task.spec_i)
    progress_state.dir_i = int(task.dir_i)
    progress_state.rho_i = int(task.rho_i)
    progress_state.cstrain_i = int(task.cstrain_i) + 1


def checkpoint_if_needed(runtime: RunRuntimeState) -> bool:
    if not runtime.tracker.should_checkpoint():
        return False
    print(f"[CHECKPOINT] written={runtime.tracker.state.written}, last_trial={runtime.tracker.state.last_trial_id} -> flush+fsync+save")
    runtime.writer.flush_fsync()
    runtime.tracker.save()
    return True


def update_runtime_progress(runtime: RunRuntimeState, task: TrialTask, evaluation: TrialEvaluation, config: ScanConfig) -> None:
    """按已完成 trial 推进 progress_state、sg_counter 和 checkpoint_state。"""
    runtime.sg_counter[int(evaluation.sg)] = runtime.sg_counter.get(int(evaluation.sg), 0) + 1
    print_phase_progress(task, evaluation, config)
    update_progress_cursor(runtime.tracker.state, task)
    sync_tracker_state(runtime)
    checkpoint_if_needed(runtime)


def flush_and_save_phase_boundary(runtime: RunRuntimeState, next_phase: str) -> None:
    """phase 切换点强制落盘，确保 single 完成后恢复时能从 combo 开始。"""
    runtime.tracker.state.phase = str(next_phase)
    if next_phase == "combo":
        runtime.tracker.state.spec_i = 0
        runtime.tracker.state.dir_i = 0
        runtime.tracker.state.rho_i = 0
        runtime.tracker.state.cstrain_i = 0
    sync_tracker_state(runtime)
    runtime.writer.flush_fsync()
    runtime.tracker.save()


def close_runtime_state(runtime: RunRuntimeState) -> None:
    if runtime.closed:
        return
    sync_tracker_state(runtime)
    runtime.writer.close()
    runtime.tracker.save()
    runtime.closed = True


def load_or_initialize_runtime_state(
    paths: RunPaths,
    config: ScanConfig,
    derived: DerivedScanConfig,
    debug: DebugOptions,
) -> RunRuntimeState:
    tracker = ProgressTracker(paths.run_dir, every=int(config.flush_every_n_records))
    ckpt = tracker.load_if_exists()

    writer = ResultWriter(
        paths.run_dir,
        tracker=tracker,
        write_csv=bool(config.write_results_csv),
        flush_every=int(config.flush_every_n_records),
    )

    if ckpt is not None:
        print(f"[RESUME] checkpoint found: phase={ckpt.phase}, written={ckpt.written}, last_trial_id={ckpt.last_trial_id}")
        trial_id = int(ckpt.last_trial_id)
        writer.n_written = int(ckpt.written)
        write_counts = {int(k): int(v) for k, v in ckpt.write_counts.items()}
        sg_counter = {int(k): int(v) for k, v in ckpt.sg_counter.items()}
    else:
        trial_id = 0
        write_counts = {}
        sg_counter = {}

    n_workers = resolve_n_workers(config, debug)
    batch_size = resolve_parallel_batch_size(config, n_workers)
    use_parallel = n_workers > 1

    print(f"[INFO] parallel workers = {n_workers} (enabled={use_parallel}, batch_size={batch_size}, chunksize={config.parallel_map_chunksize})")

    runtime = RunRuntimeState(
        tracker=tracker,
        writer=writer,
        trial_id=trial_id,
        write_counts=write_counts,
        sg_counter=sg_counter,
        n_workers=n_workers,
        batch_size=batch_size,
        use_parallel=use_parallel,
        processed_new_trials=0,
        remaining_debug_trials=debug.max_trials,
    )
    sync_tracker_state(runtime)
    return runtime


def initialize_scan_runtime(
    paths: RunPaths,
    plan: ScanPlan,
    config: ScanConfig,
    derived: DerivedScanConfig,
    debug: DebugOptions,
) -> RunRuntimeState:
    """打开 writer/checkpoint 并初始化运行态。

    这里属于主进程持久化准备阶段；调试时重点观察 tracker.state、trial_id、
    write_counts、n_workers 和 batch_size。
    """
    debug_stage_event("single", "prepare", debug, {
        "resume_phase_hint": "runtime initialization",
        "single_total_tasks": plan.single_phase.total_tasks,
        "combo_total_tasks": plan.combo_phase.total_tasks,
    })
    return load_or_initialize_runtime_state(paths, config, derived, debug)


def finalize_run_summary(
    runtime: RunRuntimeState,
    paths: RunPaths,
    config: ScanConfig,
) -> RunSummary:
    """关闭 writer/checkpoint，并生成最终摘要对象。"""
    close_runtime_state(runtime)
    return RunSummary(
        total_records=int(runtime.writer.n_written),
        sg_counter=dict(runtime.sg_counter),
        jsonl=str(runtime.writer.jsonl_path),
        csv=str(runtime.writer.csv_path) if config.write_results_csv else None,
        checkpoint=str((paths.run_dir / "checkpoint.json").resolve()),
    )

def resolve_n_workers(config: ScanConfig, debug: DebugOptions) -> int:
    if debug.no_parallel:
        return 1
    if config.n_workers is not None:
        return max(1, int(config.n_workers))

    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "OMP_NUM_THREADS"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        match = re.match(r"(\d+)", raw)
        if match:
            return max(1, int(match.group(1)))

    return max(1, os.cpu_count() or 1)


def resolve_parallel_batch_size(config: ScanConfig, n_workers: int) -> int:
    if config.parallel_batch_size is not None:
        return max(1, int(config.parallel_batch_size))
    return max(1, int(n_workers))
