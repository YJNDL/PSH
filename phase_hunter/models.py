# -*- coding: utf-8 -*-
"""核心数据结构。

本模块只放 dataclass 和轻量容器，不 import 业务模块，避免形成隐式执行路径。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

@dataclass(frozen=True)
class DebugOptions:
    """仅用于本地调试的开关；默认全部关闭，不改变正常扫描行为。"""

    enabled: bool
    stop_after_stage: Optional[str]
    no_parallel: bool
    max_trials: Optional[int]
    print_config: bool
    print_plan: bool


@dataclass(frozen=True)
class FailurePolicyOptions:
    """统一失败策略解析结果。

    主路径只读这些派生布尔值，不再在各模块分散维护旧式容错开关。
    """

    name: str
    empty_combo_scan_continues: bool
    empty_high_k_modes_continue: bool
    failed_candidate_continues: bool


@dataclass(frozen=True)
class ScanConfig:
    """用户配置快照：来自文件顶部默认值，并可被少量 CLI 参数显式覆盖。"""

    parent_poscar: str
    output_dir: str
    save_policy: str
    max_structures_per_sg: Optional[int]
    create_run_subdir: bool
    run_tag: Optional[str]
    symprec_parent: float
    symprec_identify: float
    angle_tolerance: float
    scan_profile: str
    target_sgs: List[int]
    enable_strain: bool
    confirmed_combos: List[Dict[str, Any]]
    parent_point_group_hint: Optional[str]
    strict_pointgroup_match: bool
    enable_runtime_log: bool
    runtime_log_filename: str
    flush_every_n_records: int
    print_every_n_trials: int
    print_also_on_hit: bool
    n_workers: Optional[int]
    parallel_batch_size: Optional[int]
    parallel_map_chunksize: int
    result_write_in_main: bool
    write_results_csv: bool
    auto_combo_max_size: int
    auto_dir_pool_2d: List[List[float]]
    random_op_direction_seed: int
    slurm_partition: Optional[str]
    slurm_account: Optional[str]
    slurm_qos: Optional[str]
    slurm_job_name: Optional[str]
    slurm_nodes: int
    slurm_ntasks: int
    slurm_cpus_per_task: int
    slurm_mem: Optional[str]
    slurm_time: str
    slurm_output_pattern: str
    slurm_error_pattern: str
    slurm_workdir: Optional[str]
    slurm_python_cmd: str
    slurm_script_path: Optional[str]
    slurm_extra_setup_commands: List[str]
    slurm_export_env: Dict[str, str]
    slurm_sbatch_extra_lines: List[str]
    mode_block_tol: float
    structure_dimensionality: str
    high_symmetry_kpoint_selection: str
    high_symmetry_kpoint_labels: Optional[List[str]]
    include_gamma_high_symmetry: bool
    extra_target_kpoints_fractional: Optional[List[List[float]]]
    extra_target_kpoint_labels: Optional[List[str]]
    extra_target_kpoints_basis: str
    kpoint_rationalize_max_den: int
    kpoint_tol: float
    high_symmetry_max_supercell_size: int
    real_mode_strategy: str
    failure_policy: str
    reduce_distorted_output_cell: bool
    reduce_distorted_symprec: float
    reduce_distorted_angle_tolerance: float
    reduce_distorted_min_volume_ratio: float
    reduce_distorted_verify: bool
    reduce_distorted_verify_tol: float
    reduce_distorted_keep_unreduced_copy: bool
    reduce_distorted_strict: bool
    reduce_distorted_skip_if_amplitude_below: float
    reduce_distorted_max_atoms: Optional[int]
    reduce_distorted_log_mapping: bool


@dataclass(frozen=True)
class DerivedScanConfig:
    """派生配置：扫描网格、写盘策略和目标空间群集合在这里集中解析。"""

    amp_grid: List[float]
    rho_grid: List[float]
    strain_a_grid: List[float]
    strain_b_grid: List[float]
    strain_c_grid: List[float]
    enable_random_op_directions: bool
    n_random_op_directions_per_combo: int
    random_op_direction_candidates: int
    write_all_trial_structures: bool
    write_hit_structures: bool
    write_per_sg_structures: bool
    write_limit_per_sg: Optional[int]
    target_sg_set: Set[int]
    failure_options: FailurePolicyOptions
    config_warnings: List[str]


@dataclass(frozen=True)
class Crystal:
    lattice: np.ndarray
    frac: np.ndarray
    numbers: np.ndarray
    symbols: List[str]

    @property
    def nsites(self) -> int:
        return int(self.frac.shape[0])

    def to_spglib_cell(self):
        return (np.array(self.lattice, float), np.array(self.frac, float), np.array(self.numbers, int))


@dataclass(frozen=True)
class HighSymmetryKPoint:
    """能带路径后端返回的高对称 k 点。

    k_fractional 属于 backend primitive reciprocal basis；它只是 k 点标签，
    不等同于任何外部标准表示名称。
    """

    label: str
    k_fractional: np.ndarray
    basis: str
    source_backend: str
    path_convention: str
    is_path_endpoint: bool
    path_segments: List[Tuple[str, str]]
    rationalized_k: Optional[np.ndarray]
    denominator: Optional[int]
    is_commensurate: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HighSymmetryKPathData:
    """高对称 k 路径数据；只描述 primitive BZ 中的 special points/path。"""

    backend: str
    path_convention: str
    primitive_crystal: Crystal
    point_coords: Dict[str, np.ndarray]
    path: List[Tuple[str, str]]
    selected_kpoints: List[HighSymmetryKPoint]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrimitiveCellMapping:
    """input/conventional cell 与 backend primitive cell 的原子和 k 基底映射。"""

    input_crystal: Crystal
    primitive_crystal: Crystal
    input_lattice: np.ndarray
    primitive_lattice: np.ndarray
    input_atom_to_primitive_atom: np.ndarray
    input_atom_to_primitive_lattice_shift: np.ndarray
    primitive_atom_to_input_atom: np.ndarray
    max_atom_mapping_error: float
    max_k_roundtrip_error: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KStarData:
    """某个 primitive k 点在空间群旋转下的 star 和 little group 信息。"""

    high_symmetry_label: str
    representative_k: np.ndarray
    arms: List[np.ndarray]
    arm_labels: List[str]
    little_group_indices: List[int]
    star_operation_indices: List[int]
    little_group_size: int
    star_size: int
    is_gamma: bool
    has_minus_k_in_star: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunPaths:
    output_root: Path
    run_dir: Path
    log_path: Optional[Path]


@dataclass(frozen=True)
class ParentStructureData:
    """父相结构阶段产物：保留 POSCAR 路径和解析后的 Crystal。"""

    poscar_path: Path
    crystal: Crystal


@dataclass(frozen=True)
class ParentSymmetryData:
    """对称性阶段产物：spglib 识别结果、实际使用的点群和全部对称操作。"""

    spacegroup_number: int
    spacegroup_symbol: Optional[str]
    point_group_raw: Optional[str]
    point_group_normalized: Optional[str]
    point_group_hint: Optional[str]
    point_group_used: str
    rotations: np.ndarray
    translations: np.ndarray
    strict_pointgroup_match: bool


@dataclass(frozen=True)
class ModeCellData:
    """某组模式实际作用的结构 cell。

    parent Γ 模式默认使用父相 cell；high-symmetry k 模式可能已经实化到 commensurate
    supercell，下游生成 trial 结构时必须使用这里记录的 cell。
    """

    cell_kind: str
    crystal: Crystal
    lattice: np.ndarray
    frac: np.ndarray
    numbers: np.ndarray
    symbols: List[str]
    supercell_matrix: Optional[np.ndarray]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModeBlockMetadata:
    """模式 block 的主流程元数据。

    这里记录的是内部 mode block ID 和构造来源，不是外部数据库中的标准表示标签。
    标准命名层当前不参与模式基构建。
    """

    key: str
    source: str
    source_kind: str
    sector_label: Optional[str]
    high_symmetry_label: Optional[str]
    k_vector: Optional[np.ndarray]
    k_basis: Optional[str]
    arm_index: Optional[int]
    star_size: Optional[int]
    little_group_size: Optional[int]
    block_index: int
    block_dimension: int
    mode_count: int
    is_gamma: bool
    cell_kind: str
    supercell_matrix: Optional[np.ndarray]
    phase_convention: Optional[str]
    realification_strategy: Optional[str]
    label_status: str = "not_assigned"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KPointModeMetadata:
    """高对称 k 点模式元数据；高对称 k 标签与外部标准表示名称明确分离。"""

    internal_mode_key: str
    high_symmetry_label: str
    original_label: str
    k_fractional: np.ndarray
    k_basis: str
    k_star: List[np.ndarray]
    little_group_size: int
    star_size: int
    block_index: int
    block_dimension: int
    complex_basis_available: bool
    real_supercell_available: bool
    mode_cell_key: Optional[str]
    phase_convention: str
    realification_strategy: str
    label_status: str = "not_assigned"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComplexModeBlock:
    """primitive-k Bloch mechanical representation 的 complex block。"""

    high_symmetry_label: str
    k_vector: np.ndarray
    little_group_indices: List[int]
    basis_complex: np.ndarray
    dimension_complex: int
    character: np.ndarray
    block_index: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommensurateSupercell:
    """用于把 commensurate complex k 模式实化的 supercell。"""

    primitive_crystal: Crystal
    supercell_crystal: Crystal
    supercell_matrix: np.ndarray
    det: int
    primitive_translations: List[np.ndarray]
    supercell_atom_to_primitive_atom: np.ndarray
    supercell_atom_to_primitive_lattice_shift: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LandauBasisData:
    """Landau-like basis 阶段产物：按内部 mode key 分组的对称适配位移模式。"""

    landau_basis: Dict[str, List[np.ndarray]]
    mode_keys: List[str]
    mode_key_aliases: Dict[str, str] = field(default_factory=dict)
    mode_metadata: Dict[str, ModeBlockMetadata] = field(default_factory=dict)
    mode_cell_by_key: Dict[str, ModeCellData] = field(default_factory=dict)
    kpoint_metadata: Dict[str, KPointModeMetadata] = field(default_factory=dict)


@dataclass(frozen=True)
class ComboPlanSpec:
    spec_index: int
    mode_indices: List[Tuple[str, int]]
    combo_target_sgs: Optional[List[int]]
    directions: List[List[float]]


@dataclass(frozen=True)
class SinglePhasePlan:
    """单模 phase 的显式计划：模式、振幅、strain 网格和总任务数。"""

    mode_keys: List[str]
    basis_sizes: Dict[str, int]
    amp_grid: List[float]
    single_strains: List[Tuple[float, float, float]]
    total_tasks: int


@dataclass(frozen=True)
class ComboPhasePlan:
    """多模组合 phase 的显式计划：combo specs、OP 方向、rho/strain 网格。"""

    specs: List[ComboPlanSpec]
    rho_grid: List[float]
    combo_strains: List[Tuple[float, float, float]]
    using_confirmed: bool
    total_tasks: int


@dataclass(frozen=True)
class ScanPlan:
    single_phase: SinglePhasePlan
    combo_phase: ComboPhasePlan


@dataclass(frozen=True)
class TrialEvaluationContext:
    """worker 只读上下文：trial 评估需要的父相、模式基和识别参数。"""

    parent: Crystal
    landau_basis: Dict[str, List[np.ndarray]]
    symprec_identify: float
    angle_tolerance: float
    target_sg_set: Set[int]
    mode_key_aliases: Dict[str, str] = field(default_factory=dict)
    mode_cell_by_key: Dict[str, ModeCellData] = field(default_factory=dict)


@dataclass(frozen=True)
class SingleTrialTask:
    mode_block_i: int
    mode_i: int
    amp_i: int
    strain_i: int
    mode_key: str
    mode_idx: int
    amp: float
    sa: float
    sb: float
    sc: float
    n_mode_blocks: int
    n_modes: int
    n_amps: int
    n_strains: int
    trial_id: Optional[int] = None
    phase: str = "single"


@dataclass(frozen=True)
class ComboTrialTask:
    spec_i: int
    dir_i: int
    rho_i: int
    cstrain_i: int
    mode_indices: List[Tuple[str, int]]
    op_direction: List[float]
    rho: float
    sa: float
    sb: float
    sc: float
    combo_target_sgs: Optional[List[int]]
    n_specs: int
    n_dirs: int
    n_rhos: int
    n_cstrains: int
    trial_id: Optional[int] = None
    phase: str = "combo"


TrialTask = Union[SingleTrialTask, ComboTrialTask]


@dataclass(frozen=True)
class TrialEvaluation:
    """trial 的纯计算结果；这里还没有写盘路径，也不修改 checkpoint。"""

    kind: str
    label: str
    trial_crystal: Crystal
    sg: int
    sg_sym: str
    pg: Optional[str]
    hit: bool
    mode_indices: List[Tuple[str, int]]
    op_direction: Optional[List[float]]
    amps: List[float]
    sa: float
    sb: float
    sc: float
    combo_target_sgs: Optional[List[int]]
    displacement: Optional[np.ndarray] = None
    source_mode_cell_kind: Optional[str] = None
    source_supercell_matrix: Optional[np.ndarray] = None


@dataclass(frozen=True)
class PersistResult:
    poscar_path: Optional[str]
    reduced_structure: Optional["ReducedStructureData"] = None
    output_cell_kind: str = "unreduced_supercell"
    unreduced_poscar_path: Optional[str] = None
    metadata_path: Optional[str] = None


@dataclass(frozen=True)
class ReducedStructureData:
    """畸变结构输出 cell reduction 的诊断数据。

    reduction 只尝试寻找“畸变后结构自身”的 primitive cell；它不把非 Γ 畸变结构
    强行投影回父相 primitive cell。
    """

    reduction_attempted: bool
    reduction_successful: bool
    backend: str
    original_nsites: int
    reduced_nsites: int
    original_volume: float
    reduced_volume: float
    volume_ratio: float
    original_lattice: np.ndarray
    reduced_lattice: Optional[np.ndarray]
    original_frac: np.ndarray
    reduced_frac: Optional[np.ndarray]
    reduced_numbers: Optional[np.ndarray]
    reduced_symbols: Optional[List[str]]
    symprec: float
    angle_tolerance: float
    max_reconstruction_error: Optional[float]
    formula_preserved: bool
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialRecord:
    """JSONL/CSV 记录对象；字段顺序和名称保持与历史 schema 兼容。"""

    trial: int
    trial_type: str
    mode_indices: List[Tuple[str, int]]
    op_direction: Optional[List[float]]
    amps: List[float]
    sa: float
    sb: float
    sc: float
    sg: int
    sg_sym: str
    pg: Optional[str]
    hit: bool
    label: str
    poscar: Optional[str]
    combo_target_sgs: Optional[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        # 字段顺序与 config.RECORD_SCHEMA 保持一致，避免 models.py 反向 import config.py。
        return {
            "trial": int(self.trial),
            "type": str(self.trial_type),
            "mode_indices": self.mode_indices,
            "op_direction": self.op_direction,
            "amps": self.amps,
            "sa": float(self.sa),
            "sb": float(self.sb),
            "sc": float(self.sc),
            "sg": int(self.sg),
            "sg_sym": str(self.sg_sym),
            "pg": self.pg,
            "hit": bool(self.hit),
            "label": str(self.label),
            "poscar": self.poscar,
            "combo_target_sgs": self.combo_target_sgs,
        }


@dataclass(frozen=True)
class PhaseExecutionSummary:
    phase: str
    tasks_processed: int
    start_trial_id: int
    end_trial_id: int
    completed: bool
    stopped_by_debug_limit: bool = False
    stopped_after_stage: bool = False
    skipped: bool = False
    note: Optional[str] = None


@dataclass(frozen=True)
class ScanExecutionResult:
    single_phase: PhaseExecutionSummary
    combo_phase: PhaseExecutionSummary


@dataclass(frozen=True)
class RunSummary:
    total_records: int
    sg_counter: Dict[int, int]
    jsonl: str
    csv: Optional[str]
    checkpoint: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": int(self.total_records),
            "sg_counter": dict(self.sg_counter),
            "jsonl": self.jsonl,
            "csv": self.csv,
            "checkpoint": self.checkpoint,
        }


@dataclass
class ProgressState:
    phase: str = "single"  # "single" or "combo"

    # single phase 的恢复游标：对应 mode_key -> mode -> amplitude -> strain 嵌套循环。
    mode_block_i: int = 0
    mode_i: int = 0
    amp_i: int = 0
    strain_i: int = 0

    # combo phase 的恢复游标：对应 spec -> direction -> rho -> strain 嵌套循环。
    spec_i: int = 0
    dir_i: int = 0
    rho_i: int = 0
    cstrain_i: int = 0

    written: int = 0
    last_trial_id: int = 0
    write_counts: Dict[str, int] = field(default_factory=dict)
    sg_counter: Dict[str, int] = field(default_factory=dict)


@dataclass
class RunRuntimeState:
    """运行态集中容器。

    writer_state、progress_state、checkpoint_state 都通过 tracker/writer 和显式计数器
    统一维护；主循环只在少数入口更新这些状态，避免深层函数隐式跳变。
    """

    tracker: Any
    writer: Any
    trial_id: int
    write_counts: Dict[int, int]
    sg_counter: Dict[int, int]
    n_workers: int
    batch_size: int
    use_parallel: bool
    processed_new_trials: int = 0
    remaining_debug_trials: Optional[int] = None
    closed: bool = False
