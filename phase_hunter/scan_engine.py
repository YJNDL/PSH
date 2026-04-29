# -*- coding: utf-8 -*-
"""phase 调度和 trial 主循环。

run_phase_hunter.py 显式展开 pipeline/stage 编排；本模块负责 single/combo phase 执行、
trial_id 分配、worker 调用以及主进程写盘/进度/checkpoint 的顺序推进。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from itertools import islice
from typing import Iterator, List, Optional

from .config import failure_policy_warning
from .debug_tools import should_stop_after_stage
from .models import (
    ComboPhasePlan,
    DebugOptions,
    DerivedScanConfig,
    LandauBasisData,
    PhaseExecutionSummary,
    RunPaths,
    RunRuntimeState,
    ScanConfig,
    ScanExecutionResult,
    ScanPlan,
    SinglePhasePlan,
    TrialEvaluation,
    TrialEvaluationContext,
    TrialTask,
)
from .persistence import (
    build_trial_record,
    flush_and_save_phase_boundary,
    persist_trial_artifacts,
    update_runtime_progress,
    write_trial_record,
)
from .planning import iter_combo_trial_tasks, iter_single_trial_tasks
from .trial_eval import (
    _evaluate_trial_task_from_worker,
    _init_parallel_worker,
    evaluate_trial_task,
)


def take_task_batch(task_iter: Iterator[TrialTask], batch_size: int, remaining_debug_trials: Optional[int]) -> List[TrialTask]:
    effective_batch_size = int(batch_size)
    if remaining_debug_trials is not None:
        effective_batch_size = min(effective_batch_size, max(0, int(remaining_debug_trials)))
    if effective_batch_size <= 0:
        return []
    return list(islice(task_iter, effective_batch_size))


def assign_trial_ids(task_batch: List[TrialTask], runtime: RunRuntimeState) -> List[TrialTask]:
    assigned: List[TrialTask] = []
    for task in task_batch:
        runtime.trial_id += 1
        assigned.append(replace(task, trial_id=int(runtime.trial_id)))
    return assigned


def execute_trial_batch(
    task_batch: List[TrialTask],
    trial_context: TrialEvaluationContext,
    config: ScanConfig,
    pool: Optional[ProcessPoolExecutor],
) -> List[TrialEvaluation]:
    if not task_batch:
        return []
    if pool is None:
        return [evaluate_trial_task(task, trial_context) for task in task_batch]
    return list(pool.map(_evaluate_trial_task_from_worker, task_batch, chunksize=max(1, int(config.parallel_map_chunksize))))


def run_trial_main_loop(
    *,
    phase_name: str,
    task_iter: Iterator[TrialTask],
    runtime: RunRuntimeState,
    trial_context: TrialEvaluationContext,
    paths: RunPaths,
    config: ScanConfig,
    derived: DerivedScanConfig,
    pool: Optional[ProcessPoolExecutor],
) -> PhaseExecutionSummary:
    """trial 级别主循环。

    数据流固定为：
    task_batch -> trial_task -> evaluation -> persist_result -> record -> progress/checkpoint。
    并行 worker 只返回 evaluation；主进程在这里统一完成写盘、进度更新和 checkpoint。
    """
    start_trial_id = int(runtime.trial_id) + 1
    tasks_processed = 0
    skipped_failed_candidates = 0
    end_trial_id = int(runtime.trial_id)

    while True:
        if runtime.remaining_debug_trials is not None and runtime.remaining_debug_trials <= 0:
            return PhaseExecutionSummary(
                phase=phase_name,
                tasks_processed=tasks_processed,
                start_trial_id=start_trial_id,
                end_trial_id=end_trial_id,
                completed=False,
                stopped_by_debug_limit=True,
                note="Stopped by --debug-max-trials.",
            )

        task_batch = take_task_batch(task_iter, runtime.batch_size, runtime.remaining_debug_trials)
        if not task_batch:
            note = None
            if skipped_failed_candidates:
                note = f"Skipped {skipped_failed_candidates} failed candidates under failure_policy={config.failure_policy}."
            return PhaseExecutionSummary(
                phase=phase_name,
                tasks_processed=tasks_processed,
                start_trial_id=start_trial_id,
                end_trial_id=end_trial_id,
                completed=True,
                note=note,
            )

        task_batch = assign_trial_ids(task_batch, runtime)
        try:
            evaluations = execute_trial_batch(task_batch, trial_context, config, pool)
        except Exception as exc:
            if not derived.failure_options.failed_candidate_continues:
                raise RuntimeError(f"{phase_name} trial batch failed under failure_policy='strict'.") from exc
            skipped_failed_candidates += len(task_batch)
            end_trial_id = int(task_batch[-1].trial_id) if task_batch[-1].trial_id is not None else end_trial_id
            if runtime.remaining_debug_trials is not None:
                runtime.remaining_debug_trials = max(0, runtime.remaining_debug_trials - len(task_batch))
            failure_policy_warning(
                config,
                f"Skipped failed {phase_name} batch with {len(task_batch)} candidates: {exc}",
            )
            continue

        for task, evaluation in zip(task_batch, evaluations):
            try:
                persist_result = persist_trial_artifacts(
                    task=task,
                    evaluation=evaluation,
                    run_dir=paths.run_dir,
                    write_counts=runtime.write_counts,
                    derived=derived,
                    result_write_enabled=bool(config.result_write_in_main),
                    config=config,
                )
                record = build_trial_record(task, evaluation, persist_result)
                write_trial_record(record, runtime.writer)
                update_runtime_progress(runtime, task, evaluation, config)
                runtime.processed_new_trials += 1
            except Exception as exc:
                if not derived.failure_options.failed_candidate_continues:
                    raise RuntimeError(
                        f"{phase_name} candidate trial_id={task.trial_id} failed under failure_policy='strict'."
                    ) from exc
                skipped_failed_candidates += 1
                failure_policy_warning(
                    config,
                    f"Skipped failed {phase_name} candidate trial_id={task.trial_id}: {exc}",
                )
                if runtime.remaining_debug_trials is not None:
                    runtime.remaining_debug_trials -= 1
                end_trial_id = int(task.trial_id) if task.trial_id is not None else end_trial_id
                continue

            if runtime.remaining_debug_trials is not None:
                runtime.remaining_debug_trials -= 1

            tasks_processed += 1
            end_trial_id = int(task.trial_id) if task.trial_id is not None else end_trial_id

            if runtime.remaining_debug_trials is not None and runtime.remaining_debug_trials <= 0:
                return PhaseExecutionSummary(
                    phase=phase_name,
                    tasks_processed=tasks_processed,
                    start_trial_id=start_trial_id,
                    end_trial_id=end_trial_id,
                    completed=False,
                    stopped_by_debug_limit=True,
                    note="Stopped by --debug-max-trials.",
                )


def execute_phase_tasks(
    *,
    phase_name: str,
    task_iter: Iterator[TrialTask],
    runtime: RunRuntimeState,
    trial_context: TrialEvaluationContext,
    paths: RunPaths,
    config: ScanConfig,
    derived: DerivedScanConfig,
    pool: Optional[ProcessPoolExecutor],
) -> PhaseExecutionSummary:
    """Compat wrapper: historical name for run_trial_main_loop()."""
    return run_trial_main_loop(
        phase_name=phase_name,
        task_iter=task_iter,
        runtime=runtime,
        trial_context=trial_context,
        paths=paths,
        config=config,
        derived=derived,
        pool=pool,
    )


def execute_single_phase(
    phase_plan: SinglePhasePlan,
    landau_data: LandauBasisData,
    runtime: RunRuntimeState,
    trial_context: TrialEvaluationContext,
    paths: RunPaths,
    config: ScanConfig,
    derived: DerivedScanConfig,
    pool: Optional[ProcessPoolExecutor],
) -> PhaseExecutionSummary:
    """执行 single phase：phase_plan 提供单模任务顺序，runtime 保存 trial/checkpoint 状态。"""
    print("================= 单模扫描 (single) =================")
    print(f"[INFO] mode blocks = {phase_plan.mode_keys}")
    print("====================================================")
    task_iter = iter_single_trial_tasks(phase_plan, landau_data, runtime.tracker.state)
    return run_trial_main_loop(
        phase_name="single",
        task_iter=task_iter,
        runtime=runtime,
        trial_context=trial_context,
        paths=paths,
        config=config,
        derived=derived,
        pool=pool,
    )


def execute_combo_phase(
    phase_plan: ComboPhasePlan,
    runtime: RunRuntimeState,
    trial_context: TrialEvaluationContext,
    paths: RunPaths,
    config: ScanConfig,
    derived: DerivedScanConfig,
    pool: Optional[ProcessPoolExecutor],
) -> PhaseExecutionSummary:
    """执行 combo phase：phase_plan 提供多模组合任务，主进程继续沿用同一 runtime。"""
    print()
    print("================= 多模组合扫描 (combos) =================")
    print(f"[INFO] combo_specs 数量 = {len(phase_plan.specs)} (using_confirmed={phase_plan.using_confirmed})")
    print(f"[INFO] random_op_dirs = {derived.enable_random_op_directions} (n_random={derived.n_random_op_directions_per_combo})")
    print("=========================================================")
    task_iter = iter_combo_trial_tasks(phase_plan, runtime.tracker.state)
    return run_trial_main_loop(
        phase_name="combo",
        task_iter=task_iter,
        runtime=runtime,
        trial_context=trial_context,
        paths=paths,
        config=config,
        derived=derived,
        pool=pool,
    )


def execute_scan_plan(
    plan: ScanPlan,
    runtime: RunRuntimeState,
    trial_context: TrialEvaluationContext,
    landau_data: LandauBasisData,
    paths: RunPaths,
    config: ScanConfig,
    derived: DerivedScanConfig,
    debug: DebugOptions,
) -> ScanExecutionResult:
    """按 checkpoint 状态顺序执行 single/combo 两个 phase。"""
    single_summary = PhaseExecutionSummary(
        phase="single",
        tasks_processed=0,
        start_trial_id=int(runtime.trial_id) + 1,
        end_trial_id=int(runtime.trial_id),
        completed=False,
        skipped=True,
        note="Single phase not entered.",
    )
    combo_summary = PhaseExecutionSummary(
        phase="combo",
        tasks_processed=0,
        start_trial_id=int(runtime.trial_id) + 1,
        end_trial_id=int(runtime.trial_id),
        completed=False,
        skipped=True,
        note="Combo phase not entered.",
    )

    ctx = (
        ProcessPoolExecutor(
            max_workers=runtime.n_workers,
            initializer=_init_parallel_worker,
            initargs=(trial_context,),
        )
        if runtime.use_parallel else nullcontext()
    )

    with ctx as pool:
        pool_obj = pool if runtime.use_parallel else None

        if runtime.tracker.state.phase == "single":
            single_summary = execute_single_phase(
                plan.single_phase,
                landau_data,
                runtime,
                trial_context,
                paths,
                config,
                derived,
                pool_obj,
            )
            if single_summary.completed:
                flush_and_save_phase_boundary(runtime, next_phase="combo")
                print("[INFO] single 扫描完成，切换到 combo 扫描。")
                if should_stop_after_stage(debug, "single"):
                    single_summary = replace(single_summary, stopped_after_stage=True, note="Stopped after single stage.")
                    return ScanExecutionResult(single_phase=single_summary, combo_phase=combo_summary)
            else:
                return ScanExecutionResult(single_phase=single_summary, combo_phase=combo_summary)
        else:
            single_summary = PhaseExecutionSummary(
                phase="single",
                tasks_processed=0,
                start_trial_id=int(runtime.trial_id) + 1,
                end_trial_id=int(runtime.trial_id),
                completed=False,
                skipped=True,
                note="Checkpoint already resumed at combo phase.",
            )
            if should_stop_after_stage(debug, "single"):
                single_summary = replace(single_summary, stopped_after_stage=True)
                return ScanExecutionResult(single_phase=single_summary, combo_phase=combo_summary)

        if not plan.combo_phase.specs:
            if not derived.failure_options.empty_combo_scan_continues:
                raise RuntimeError(
                    "No combo specs were generated. Fail-fast mode does not allow silently ending before combo phase."
                )
            failure_policy_warning(config, "No combo specs were generated; skipping combo phase explicitly.")
            combo_summary = PhaseExecutionSummary(
                phase="combo",
                tasks_processed=0,
                start_trial_id=int(runtime.trial_id) + 1,
                end_trial_id=int(runtime.trial_id),
                completed=False,
                skipped=True,
                note="combo_specs is empty.",
            )
            return ScanExecutionResult(single_phase=single_summary, combo_phase=combo_summary)

        combo_summary = execute_combo_phase(
            plan.combo_phase,
            runtime,
            trial_context,
            paths,
            config,
            derived,
            pool_obj,
        )
        if combo_summary.completed and should_stop_after_stage(debug, "combo"):
            combo_summary = replace(combo_summary, stopped_after_stage=True, note="Stopped after combo stage.")
        return ScanExecutionResult(single_phase=single_summary, combo_phase=combo_summary)
