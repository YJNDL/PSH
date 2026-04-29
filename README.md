# PSH (Phase Hunter)

PSH 是一个用于晶体结构相变/对称性扫描的 Python 工具集，主入口为 `run_phase_hunter.py`。

## 快速开始

```bash
python run_phase_hunter.py --help
```

常见模式：

```bash
# 本地扫描
python run_phase_hunter.py

# 仅生成 SLURM 提交脚本
python run_phase_hunter.py --write-slurm

# 生成并立即提交 SLURM 脚本
python run_phase_hunter.py --submit-slurm
```

## 依赖

主流程依赖：

- numpy
- scipy
- spglib
- seekpath

建议在独立虚拟环境中安装依赖后再运行。

## 关键参数说明

### 其他常用参数

- `--parent-poscar`: 覆盖默认父相 POSCAR 路径。
- `--output-dir`: 覆盖默认输出目录。
- `--structure-dimensionality {2d,3d}`: 控制高对称 k 点筛选维度。
- `--high-symmetry-kpoint-selection {path_endpoints,all_point_coords,labels}`: 控制高对称 k 点来源。
- `--debug`: 打开调试日志。
- `--debug-stop-after-stage <stage>`: 在指定阶段安全退出。

## 提示

- 本仓库当前保留了入口脚本 + `phase_hunter/` 模块化实现。
- 对于批量任务，优先使用 `--write-slurm`/`--submit-slurm` 走集群流程。
