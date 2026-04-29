#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase Hunter 可执行入口。

这个入口文件按“主程序开始”的方式显式写出运行阶段，方便在 PyCharm 中直接顺着变量
和函数调用看逻辑、下断点。科学算法仍在 `phase_hunter/` 各模块中，入口这里只做调度。

主扫描阶段：
    args
        -> config / derived / debug
        -> parent_data
        -> symmetry
        -> paths
        -> landau_data
        -> plan
        -> runtime
        -> trial_context
        -> execution
        -> summary

trial 主循环在 `execute_scan_plan(...)` 内进入：
    execute_scan_plan(...)
        -> execute_single_phase(...) / execute_combo_phase(...)
        -> run_trial_main_loop(...)

主进程/worker 分工：
    worker 只计算 TrialEvaluation；
    JSONL/CSV/POSCAR/checkpoint 写入只在主进程完成。
"""

from __future__ import annotations

from dataclasses import asdict

from phase_hunter.cli import parse_cli_args
from phase_hunter.config import (
    PROGRAM_NAME,
    PROGRAM_TAG,
    PROGRAM_VERSION,
    build_scan_config_from_defaults_and_cli,
    check_required_dependencies,
)
from phase_hunter.debug_tools import (
    debug_stage_event,
    print_effective_config,
    print_run_summary,
    print_scan_plan_summary,
    should_stop_after_stage,
)
from phase_hunter.persistence import finalize_run_summary, initialize_scan_runtime
from phase_hunter.runtime_io import (
    enable_console_line_buffering,
    load_parent_structure,
    prepare_run_paths_and_logging,
)
from phase_hunter.landau_basis import build_landau_context
from phase_hunter.planning import build_scan_plan
from phase_hunter.scan_engine import execute_scan_plan
from phase_hunter.slurm import (
    _shell_quote,
    submit_slurm_script,
    write_slurm_submit_script,
)
from phase_hunter.symmetry import resolve_parent_symmetry
from phase_hunter.trial_eval import build_trial_evaluation_context


# =============================================================================
# 主程序开始
# =============================================================================
if __name__ == "__main__":
    # 1. CLI 参数解析。
    # parse_cli_args() 只把命令行选项读成 argparse.Namespace，不读取 POSCAR、不建目录、
    # 不启动扫描。典型字段包括：
    #   args.write_slurm / args.submit_slurm：是否走 SLURM 脚本分支；
    #   args.slurm_file：可选的 SLURM 脚本输出路径；
    #   args.parent_poscar / args.output_dir：本次运行覆盖默认路径；
    #   args.debug_*：只影响本次调试运行的开关。
    # PyCharm 调试时，这里是实际运行的第一个可观察变量。
    args = parse_cli_args()

    # 2. SLURM 分支。
    # 如果命令行带 --write-slurm 或 --submit-slurm，只生成/提交 sbatch 脚本，
    # 不读取父相结构、不构造 basis、不执行任何 trial。生成的脚本仍会调用本文件
    # run_phase_hunter.py，因此集群上真正扫描时也走下面同一套显式主流程。
    if args.write_slurm or args.submit_slurm:
        # build_scan_config_from_defaults_and_cli(args)：
        #   输入 args：命令行覆盖项；
        #   输出 config：最终用户配置，用于决定 SLURM 参数和脚本里调用的入口；
        # 这里 derived/debug 暂时不用，所以用 _ 接住。
        config, _, _ = build_scan_config_from_defaults_and_cli(args)

        # args.slurm_file 是用户传入的 --slurm-file；若为 None，则 slurm.py 根据配置
        # 自动生成默认 submit_*.slurm 文件名。
        slurm_path = write_slurm_submit_script(args.slurm_file, config=config)
        print(f"[SLURM] submit script written to: {slurm_path}")
        print(f"[SLURM] run manually with: sbatch {_shell_quote(slurm_path)}")

        # --submit-slurm 比 --write-slurm 多一步：立刻调用 sbatch 提交。
        if args.submit_slurm:
            submit_slurm_script(slurm_path)

    # 3. 本地主进程扫描分支。下面按阶段显式传递数据，主流程就写在本入口文件里。
    else:
        # 让 stdout/stderr 更偏向行缓冲，长时间扫描时日志能及时写到终端/文件。
        enable_console_line_buffering()

        # 程序启动 banner，只用于人读日志，不参与任何科学计算。
        print(f"[{PROGRAM_TAG}] {PROGRAM_NAME} v{PROGRAM_VERSION}")
        print(f"[{PROGRAM_TAG}] streaming + checkpoint + progress print")
        print()

        # ------------------------------------------------------------------
        # config 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   args：命令行参数。
        # 输出：
        #   config：用户配置快照，例如 PARENT_POSCAR、TARGET_SGS、SAVE_POLICY、
        #           STRICT_POINTGROUP_MATCH、并行参数、checkpoint/flush 参数等；
        #   derived：派生配置，例如 AMP_GRID/RHO_GRID/strain grid、target_sg_set、
        #            SAVE_POLICY 解析后的写盘布尔值、随机方向数量等；
        #   debug：debug-only 配置，例如 stop_after_stage、no_parallel、max_trials。
        # 注意：
        #   config/derived/debug 后续都显式传入各阶段，避免阶段函数偷偷读取全局运行态。
        config, derived, debug = build_scan_config_from_defaults_and_cli(args)

        # debug_stage_event() 只在 --debug 打开时打印阶段信息；这里记录 config 阶段
        # 已完成，以及几个最影响扫描范围和执行方式的配置值。
        debug_stage_event("config", "end", debug, {
            "scan_profile": config.scan_profile,
            "target_sgs": config.target_sgs,
            "debug_no_parallel": debug.no_parallel,
            "debug_max_trials": debug.max_trials,
        })

        # 配置构建阶段可能产生非致命 warning，例如 SAVE_POLICY 组合或 profile 派生值提示。
        for warning in derived.config_warnings:
            print(warning)

        # --debug-print-config：打印最终生效配置和派生配置，便于确认 CLI 覆盖是否生效。
        if debug.print_config:
            print_effective_config(config, derived, debug)

        # --debug-stop-after-stage config：只检查配置阶段，不要求 POSCAR 存在。
        if should_stop_after_stage(debug, "config"):
            raise SystemExit(0)

        # fail-fast 主路径依赖检查：实际扫描固定依赖 numpy/scipy/spglib/seekpath。
        # 不再根据环境自动换 k-path backend；缺依赖时立即报错。
        check_required_dependencies()

        # ------------------------------------------------------------------
        # parent 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   config.parent_poscar：父相 POSCAR 路径，可被 --parent-poscar 覆盖；
        #   debug：只控制阶段打印，不改变结构读取语义。
        # 输出：
        #   parent_data：ParentStructureData，包含 poscar_path 和解析后的 Crystal。
        # Crystal 内部约定：
        #   lattice 三行是晶格矢量；frac 是分数坐标；numbers/symbols 是元素信息。
        parent_data = load_parent_structure(config, debug)
        if should_stop_after_stage(debug, "parent"):
            raise SystemExit(0)

        # ------------------------------------------------------------------
        # symmetry 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   parent_data.crystal：父相结构；
        #   config.symprec_parent / parent_point_group_hint /
        #   strict_pointgroup_match：父相对称性识别和点群校验参数；
        #   debug：打印识别结果。
        # 输出：
        #   symmetry：ParentSymmetryData，包含父相 SG/PG、实际使用的 point group、
        #             spglib rotations/translations，以及 STRICT_POINTGROUP_MATCH 状态。
        # 后续 build_landau_context() 会用 rotations/translations 构造位移表示矩阵。
        symmetry = resolve_parent_symmetry(parent_data, config, debug)
        if should_stop_after_stage(debug, "symmetry"):
            raise SystemExit(0)

        # ------------------------------------------------------------------
        # paths 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   config.output_dir / create_run_subdir / run_tag：决定输出根目录和 run_dir；
        #   symmetry：用于在 run_dir 命名中带上父相 SG/PG 信息；
        #   debug：控制路径阶段打印。
        # 输出：
        #   paths：RunPaths，集中保存 output_root、run_dir、log_path。
        # 副作用：
        #   创建输出目录；如启用 runtime log，会把之后的 stdout/stderr tee 到 run.log。
        paths = prepare_run_paths_and_logging(config, symmetry, debug)
        if should_stop_after_stage(debug, "paths"):
            raise SystemExit(0)

        # ------------------------------------------------------------------
        # basis 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   parent_data：父相原子、晶格；
        #   symmetry：父相对称操作和 point group；
        #   config 中的 high-symmetry k-path / supercell 配置：决定要分析哪些 primitive-BZ k 点；
        #   debug：打印模式基数量和阶段信息。
        # 输出：
        #   landau_data：LandauBasisData，包含 high-k supercell real modes、
        #                mode keys、mode_cell_by_key 和 kpoint_metadata。
        landau_data = build_landau_context(parent_data, symmetry, config, debug)
        if should_stop_after_stage(debug, "basis"):
            raise SystemExit(0)

        # ------------------------------------------------------------------
        # plan 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   parent_data / symmetry / landau_data：决定有哪些模式、模式标签和组合候选；
        #   config：combo 策略、confirmed_combos、随机种子等用户配置；
        #   derived：AMP_GRID、RHO_GRID、strain grid、随机方向数量等派生扫描网格；
        #   debug：可打印计划摘要。
        # 输出：
        #   plan：ScanPlan，包含 single_phase 和 combo_phase。
        # 注意：
        #   build_scan_plan() 只生成任务计划和顺序，不执行 trial，不写任何结果文件。
        plan = build_scan_plan(parent_data, symmetry, landau_data, config, derived, debug)

        # --debug-print-plan：在正式执行 trial 前打印任务数量，适合确认扫描规模。
        if debug.print_plan:
            print_scan_plan_summary(plan, landau_data, derived)

        # --debug-stop-after-stage plan：构造到扫描计划为止，不打开 writer，不进入 trial。
        if should_stop_after_stage(debug, "plan"):
            raise SystemExit(0)

        # ------------------------------------------------------------------
        # runtime 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   paths.run_dir：writer/checkpoint/schema 的落盘位置；
        #   plan：用于 debug banner 显示 single/combo 总任务数；
        #   config：flush_every_n_records、write_results_csv、并行默认值等；
        #   derived：SAVE_POLICY 派生结果会在后续写 POSCAR 时使用；
        #   debug：debug.max_trials 和 debug.no_parallel 会影响 runtime。
        # 输出：
        #   runtime：RunRuntimeState，主进程唯一运行态容器，包含：
        #       tracker：checkpoint/progress 管理器；
        #       writer：JSONL/CSV writer；
        #       trial_id：主进程分配的最新 trial id；
        #       write_counts / sg_counter：结构写出计数和 SG 统计；
        #       n_workers / batch_size / use_parallel：并行执行参数；
        #       remaining_debug_trials：--debug-max-trials 的剩余额度。
        # 副作用：
        #   打开 JSONL/CSV 文件；读取已有 checkpoint；必要时创建 schema.json。
        runtime = initialize_scan_runtime(paths, plan, config, derived, debug)

        # ------------------------------------------------------------------
        # trial_context 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   parent_data：父相 Crystal；
        #   landau_data：位移模式基和 mode_key alias；
        #   config：trial 结构识别所需的 symprec_identify / angle_tolerance；
        #   derived：target_sg_set，用于 hit 判定。
        # 输出：
        #   trial_context：TrialEvaluationContext，传给串行 evaluator 或并行 worker。
        # worker 边界：
        #   trial_context 是只读的；worker 只用它计算 TrialEvaluation，不接触 writer、
        #   checkpoint、paths 或 runtime。
        trial_context = build_trial_evaluation_context(parent_data, landau_data, config, derived)

        # ------------------------------------------------------------------
        # execution 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   plan：single/combo 任务生成顺序；
        #   runtime：主进程运行态，负责 trial_id、writer、checkpoint、进度；
        #   trial_context：worker/evaluator 只读上下文；
        #   landau_data：single phase 任务迭代需要 basis 尺寸和 mode_keys；
        #   paths：POSCAR/JSONL/CSV/checkpoint 的 run_dir；
        #   config/derived/debug：执行参数、写盘策略和 debug 限制。
        # 输出：
        #   execution：ScanExecutionResult，记录 single_phase/combo_phase 是否完成、
        #              处理了多少 task、是否因 debug 限制停止。
        # 内部数据流在 scan_engine.run_trial_main_loop() 中固定为：
        #   task_batch -> trial_task -> evaluation -> persist_result
        #       -> trial_record -> writer/checkpoint/progress
        execution = execute_scan_plan(plan, runtime, trial_context, landau_data, paths, config, derived, debug)

        # ------------------------------------------------------------------
        # summary 阶段
        # ------------------------------------------------------------------
        # 输入：
        #   runtime：包含 writer 计数、SG 统计和 checkpoint tracker；
        #   paths：用于定位 checkpoint.json；
        #   config：决定是否报告 CSV 路径。
        # 输出：
        #   summary：RunSummary，包含总记录数、发现的 SG 统计、JSONL/CSV/checkpoint 路径。
        # 副作用：
        #   finalize_run_summary() 会 flush/close writer 并保存 checkpoint。
        summary = finalize_run_summary(runtime, paths, config)

        # 把 execution 的 dataclass 摘要转成 dict，仅用于 debug 输出；不影响结果文件。
        debug_stage_event("summary", "end", debug, {
            "single_phase": asdict(execution.single_phase),
            "combo_phase": asdict(execution.combo_phase),
            "total_records": summary.total_records,
        })

        # 最后在人类可读日志里打印本次运行结果位置和 SG 统计。
        print_run_summary(summary)
