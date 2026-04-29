# PSH (Phase Hunter)

PSH 是一个用于晶体结构扫描与相变候选探索的 Python 工具。主入口脚本是 `run_phase_hunter.py`。

---

## 项目结构

- `run_phase_hunter.py`：完整主流程入口（推荐直接运行）
- `phase_hunter/config.py`：默认配置、CLI 覆盖与派生配置
- `phase_hunter/cli.py`：参数解析与 SLURM 分支
- `phase_hunter/scan_engine.py`：single/combo 扫描执行
- `phase_hunter/persistence.py`：结果写盘（JSONL/CSV/POSCAR/checkpoint）
- `phase_hunter/runtime_io.py`：运行目录与日志 tee

---

## 依赖

建议 Python 3.10+。

安装核心依赖：

```bash
python -m pip install numpy scipy spglib seekpath
```

---

## 快速开始

### 1) 查看帮助

```bash
python run_phase_hunter.py --help
```

### 2) 本地运行

```bash
python run_phase_hunter.py
```

### 3) SLURM 模式

```bash
# 仅生成提交脚本
python run_phase_hunter.py --write-slurm

# 生成并立即提交
python run_phase_hunter.py --submit-slurm
```

---

## 常用参数

### 输入输出

- `--parent-poscar PATH`：父相 POSCAR 路径
- `--output-dir PATH`：输出目录

### 扫描维度 / k 点

- `--structure-dimensionality {2d,3d}`
- `--high-symmetry-kpoint-selection {path_endpoints,all_point_coords,labels}`
- `--high-symmetry-kpoint-labels "K1,K2,..."`
- `--exclude-gamma-high-symmetry`

### 失败策略

- `--failure-policy {strict,debug,permissive}`

### 约化相关

- `--reduce-distorted-symprec FLOAT`
- `--reduce-distorted-skip-if-amplitude-below FLOAT`

### 调试参数

- `--debug`
- `--debug-stop-after-stage {config,paths,parent,symmetry,basis,plan,single,combo,summary}`
- `--debug-no-parallel`
- `--debug-max-trials N`
- `--debug-print-config`
- `--debug-print-plan`

---

## 输出文件说明

运行目录下典型输出：

- `scan_results.jsonl`：主结果（逐行 JSON）
- `scan_results.csv`：可选 CSV
- `checkpoint.json`：断点续跑状态
- `run.log`：运行日志（启用 runtime log 时）
- `structures_by_sg/` 或 `hit_structures_by_sg/`：按空间群分组的 POSCAR 输出
- `*.structure_metadata.json`：结构约化元信息

---

## 常见问题

### 启动时报依赖缺失

先检查依赖是否安装在当前解释器：

```bash
python -m pip show numpy scipy spglib seekpath
```

### 只想检查配置，不启动全扫描

```bash
python run_phase_hunter.py --debug --debug-stop-after-stage config
```

### 需要低 IO/快速排错

可先降低扫描规模并减少写盘：

- 选择更轻量 profile（如 `fast`）
- 关闭 CSV 写出（改 `WRITE_RESULTS_CSV=False`）
- 增大 `FLUSH_EVERY_N_RECORDS`
- 提高 `PRINT_EVERY_N_TRIALS`

