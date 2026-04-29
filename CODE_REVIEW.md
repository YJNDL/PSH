# 代码审查报告（继续，2026-04-29）

## 结论
本轮继续审查后：
- 之前的 CLI 示例入口名问题已修复；
- 仍有 1 个阻断问题 + 1 个中优先级可用性问题待处理。
当前主入口的“依赖 fail-fast”设计没有按预期生效：在给出友好依赖提示前，程序会先因模块顶层导入失败而直接中断。

---

## 已修复项（本轮）

### CLI 帮助示例入口名已统一
- `phase_hunter/cli.py` 的 argparse 示例已从 `python this_script.py ...` 改为 `python run_phase_hunter.py ...`。
- 可避免用户复制示例时使用错误入口。

---

## 问题 1（阻断）：依赖 fail-fast 检查仍被顶层导入顺序绕过

### 现象
- 主入口 `run_phase_hunter.py` 设计上会调用 `check_required_dependencies()` 做统一依赖检查。
- 但 `phase_hunter/config.py` 顶层先执行了 `import numpy as np`，导致缺少 `numpy` 时在模块导入阶段就直接崩溃，无法进入统一检查逻辑。

### 复现
```bash
python -c "import run_phase_hunter"
```

### 实际结果
- 报错：`ModuleNotFoundError: No module named 'numpy'`
- 位置：`phase_hunter/config.py` 顶层导入。

### 影响
- 用户看不到统一依赖错误提示；启动路径异常提前中断。

### 建议
1. 把 `numpy` 顶层导入改为函数内部延迟导入；或
2. 将依赖第三方库的 profile 网格生成逻辑改为纯 Python；并
3. 保证主入口先完成统一依赖检查，再进入后续模块重依赖流程。

---

## 问题 2（中优先级）：README 快速开始中的 `--help` 在缺依赖环境下不稳定

### 现象
- README 推荐 `python run_phase_hunter.py --help` 作为快速开始。
- 由于当前入口导入阶段依赖 `numpy`，在未装依赖时该命令不会稳定返回帮助信息，而会直接导入失败。

### 影响
- 新用户在“查看帮助”这一步就可能失败，降低可用性。

### 建议
- 在 README 的快速开始段落补充“先安装依赖，再执行 --help”；或
- 提供一个不触发重依赖导入的轻量帮助入口（例如独立 CLI shim）。

---

## 已执行检查
- `python -c "import run_phase_hunter"`：失败（`ModuleNotFoundError: No module named 'numpy'`）。
- `python -m py_compile phase_hunter/cli.py phase_hunter/config.py run_phase_hunter.py`：通过（语法层面正常）。
### 1) 依赖检查函数被顶层导入顺序绕过（阻断）
- `run_phase_hunter.py` 在主流程中调用 `check_required_dependencies()`，意图在扫描前统一检查 `numpy/scipy/spglib/seekpath` 并给出可读报错。
- 但 `phase_hunter/config.py` 顶层已执行 `import numpy as np`，因此只要环境缺少 `numpy`，程序会在 `import run_phase_hunter` 阶段直接抛出 `ModuleNotFoundError`，根本到不了 `check_required_dependencies()`。
- 这使得当前“集中 fail-fast + 统一提示”的实现失效。

**复现**：
- 命令：`python -c "import run_phase_hunter"`
- 结果：在 `phase_hunter/config.py` 顶层 `import numpy as np` 处报 `ModuleNotFoundError: No module named 'numpy'`。

**影响**：
- 主入口无法按设计输出统一依赖诊断；对用户来说是“启动即崩溃”，错误信息不一致且不可控。

**建议修复**：
1. 将 `config.py` 中对 `numpy` 的导入延迟到需要位置（例如网格构建函数内部）；或
2. 用纯 Python 生成 profile 网格，避免 `config.py` 的顶层第三方依赖；并
3. 保留 `check_required_dependencies()` 作为唯一统一依赖门禁（确保先进入它，再触发后续重依赖逻辑）。

## 已执行检查
- `python -c "import run_phase_hunter"`：失败（`ModuleNotFoundError: No module named 'numpy'`）。
- `git status --short`：仅本报告文件发生修改。
