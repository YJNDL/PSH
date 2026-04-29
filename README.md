# PSH / Phase Hunter

PSH（`phase_hunter`）是一个面向晶体结构扫描的工具集，用于基于父相结构、对称性与高对称 k 点模式构造 trial 结构，并执行空间群识别、结果持久化与断点续跑。

主程序入口：

- `run_phase_hunter.py`（推荐，完整本地扫描流程）
- `phase_hunter/cli.py`（仅做 CLI 解析和 SLURM 分支兼容）

---

## 1. 环境准备

建议 Python 3.10+。

安装核心依赖：

```bash
python -m pip install numpy scipy spglib seekpath
```

> 若你在集群上运行，建议使用虚拟环境或 conda 环境，避免系统 Python 依赖冲突。

---

## 2. 快速开始

查看帮助：

```bash
python run_phase_hunter.py --help
```

最小本地运行：

```bash
python run_phase_hunter.py
```

仅生成 SLURM 脚本：

```bash
python run_phase_hunter.py --write-slurm
```

生成并提交 SLURM 脚本：

```bash
python run_phase_hunter.py --submit-slurm
```

---

## 3. 常用参数

### 输入/输出

- `--parent-poscar PATH`：覆盖默认父相 POSCAR 路径。
- `--output-dir PATH`：覆盖默认输出目录。

### k 点与维度

- `--structure-dimensionality {2d,3d}`：
  - `2d`：保留 `kz=0` 的高对称点；
  - `3d`：保留完整 3D BZ 高对称点。
- `--high-symmetry-kpoint-selection {path_endpoints,all_point_coords,labels}`：选择高对称点来源。
- `--high-symmetry-kpoint-labels "A,B,C"`：当 selection=labels 时生效。
- `--exclude-gamma-high-symmetry`：不自动加入 Gamma/G。

### 失败策略

- `--failure-policy {strict,debug,permissive}`：
  - `strict`：关键路径失败即停止；
  - `debug`：允许部分调试场景继续；
  - `permissive`：批量探索时更宽松。

### 结构约化相关

- `--reduce-distorted-symprec FLOAT`：畸变结构约化识别使用的 `symprec`。
- `--reduce-distorted-skip-if-amplitude-below FLOAT`：位移幅度低于阈值时跳过约化。

### 调试参数

- `--debug`：开启调试信息。
- `--debug-stop-after-stage STAGE`：在指定阶段后安全退出（例如 `config/parent/symmetry/plan`）。
- `--debug-no-parallel`：禁用并行，便于定位问题。
- `--debug-max-trials N`：限制本次最多处理 N 个 trial。
- `--debug-print-config`：打印最终生效配置。
- `--debug-print-plan`：打印扫描计划摘要。

---

## 4. 运行输出说明

典型输出目录由 `output_dir` + 时间戳子目录构成，通常包含：

- `run.log`：运行日志（启用 runtime log 时）。
- `results.jsonl`：主结果记录（逐行 JSON）。
- `results.csv`：可选 CSV 汇总。
- `checkpoint.json`：断点续跑状态。
- `structures_by_sg/` 或 `hit_structures_by_sg/`：按空间群分类的 POSCAR 输出。
- `*.structure_metadata.json`：结构约化元信息（在约化相关功能启用时产生）。

---

## 5. 常见问题

### Q1: 程序启动报依赖缺失

请先确认当前解释器安装了：`numpy scipy spglib seekpath`。

```bash
python -m pip show numpy scipy spglib seekpath
```

### Q2: 只想验证配置，不想真正扫描

可使用：

```bash
python run_phase_hunter.py --debug --debug-stop-after-stage config
```

### Q3: 想在单进程下复现问题

可使用：

```bash
python run_phase_hunter.py --debug --debug-no-parallel --debug-max-trials 2
```

---

## 6. 开发者提示

- 入口调度逻辑主要在 `run_phase_hunter.py`。
- 配置与 CLI 覆盖在 `phase_hunter/config.py`。
- 扫描执行引擎在 `phase_hunter/scan_engine.py`。
- 结果持久化在 `phase_hunter/persistence.py`。
- 运行路径与日志配置在 `phase_hunter/runtime_io.py`。

如果你要修改写盘策略（`SAVE_POLICY/per_sg`）相关逻辑，建议优先阅读 `maybe_write_trial_structure(...)` 与 `persist_trial_artifacts(...)`。
