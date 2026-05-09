# Riskfield

本项目提供了一套用于交通场景风险评估和可视化工具，基于 `RiskFlow` 模型。它支持训练模型、评估性能、运行风险排序实验以及生成风险场的可视化视频。

## 功能特性

- **模型训练与评估**：基于 InD 数据集训练 `RiskFlow` 模型。
- **风险排序测试**：计算模型预测风险与客观风险（如基于距离、TTC 等）的相关性。
- **风险场可视化**：生成包含似然场（Likelihood Field）、风险场（Risk Field）和车辆轨迹的可视化 MP4 视频。
- **多样化的客观风险指标**：支持 Gaussian Relative Velocity, Gaussian Distance, Inverse Distance, Min Distance, TTC 等。

## 安装

确保已安装必要的依赖项：

```bash
pip install torch numpy matplotlib wandb opencv-python ffmpeg-python
```

此外，可视化功能需要系统中安装有 `ffmpeg`。

## 使用说明

### 1. 训练与基础评估

使用 `main.py` 进行模型的训练、序列化（保存）和基础评估（RMSE, CRPS, ADE, FDE 等）。

```bash
python main.py
```

*注意：可以通过修改 `main.py` 中的 `should_train`, `should_serialize`, `should_evaluate` 布尔值来控制流程。训练配置主要在 `riskflow_config.py` 中定义。*

### 2. 风险排序实验 (Risk Ranking)

用于评估模型预测的风险场与客观物理风险指标（Ground Truth Based）之间的单调相关性（Spearman Correlation）。

#### 单个/指定片段测试
使用 `risk_ranking_experiment.py` 对指定的样本索引进行详细测试。

**示例（针对 InD Site 08）：**
```bash
python risk_ranking_experiment.py --model serialized/riskflow_ind_0.pt --site 08 --indices 363 1072 --steps 64 --objective gaussian_relv --save-fields
```

#### 批量随机测试
使用 `risk_ranking_batch.py` 在测试集中随机抽取 N 个样本进行批量评估，并输出汇总统计信息。

**示例（针对 InD Site 08）：**
```bash
python risk_ranking_batch.py --model serialized/riskflow_ind_0.pt --site 08 --n 50 --min-env 2 --objective ttc --out videos/risk_ranking_batch_08
```

#### 参数详解

- **通用参数**:
  - `--model`: 必须参数。模型权重文件（`.pt`）的路径。
  - `--site`: InD 数据集地点 ID，推荐使用 `08`。
  - `--device`: 运行设备（`cpu`, `cuda`, `auto`）。
  - `--steps`: 风险场网格分辨率（例如 `64` 代表生成 64x64 的热力图网格）。
  - `--cost-step`: 采样步长，控制时间间隔 $dt = 0.04 \times cost\_step$。默认值为 `2`。

- **客观风险定义 (`--objective`)**:
  - `gaussian_relv`: (默认) 基于高斯权重的相对速度风险。
  - `gaussian_dist`: 仅考虑周围车辆的高斯距离分布。
  - `inv_dist`: 距离倒数之和。
  - `min_dist`: 最近邻车辆距离的倒数。
  - `ttc`: 基于碰撞时间 (Time-to-Collision) 的风险，考虑预测轨迹的交汇情况。

- **特定于客观风险的细化参数**:
  - `--sigma`: 高斯权重的标准差（米），默认 `1.5`。
  - `--ttc-max`: TTC 截断阈值（秒），默认 `5.0`。
  - `--ttc-tau`: TTC 衰减常数，默认 `2.0`。

- **批处理相关 (`risk_ranking_batch.py`)**:
  - `--n`: 随机采样的样本总数。
  - `--min-env`: 筛选样本的门槛，要求该片段中至少包含多少辆周围车辆（不含 Ego）。
  - `--save-per-seg`: 是否为每个片段单独保存相关性结果。

- **输出与缓存**:
  - `--out`: 结果保存目录。
  - `--cache-dir`: 似然场计算开销较大，可以使用该参数指定缓存目录以加速重复实验。

### 3. 可视化 (Visualization)

`visualize_field.py` 支持多种可视化模式。通常你会通过其他脚本调用它，或者根据需要编写简单的启动脚本。

基本可视化示例（逻辑通常集成在实验脚本中）：
- 生成**似然场**视频：展示模型预测的车辆未来位置分布。
- 生成**风险场**视频：结合自我车辆（Ego）速度与环境车辆似然场生成的风险热力图。

可视化结果通常保存在 `videos/` 目录下。

## 示例输出

- **`videos/risk_ranking_batch/summary.npz`**: 包含批量实验的 Spearman 相关性统计结果。
- **`videos/sample_{idx}/video0.mp4`**: 展示特定片段的风险场演变。

## 核心文件说明

- `main.py`: 项目入口，控制训练与评估主流程。
- `model/RiskFlow.py`: 核心模型实现。
- `risk_ranking_batch.py`: 风险相关性批量测试工具。
- `risk_ranking_experiment.py`: 风险字段计算与相关性分析核心逻辑。
- `visualize_field.py`: 基于 Matplotlib 和 FFmpeg 的可视化工具类。
- `riskflow_config.py`: 包含模型和实验的默认超参数配置。
