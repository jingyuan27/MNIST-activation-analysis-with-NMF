# MNIST 隐藏层激活分析与非负矩阵分解

通过一个手写数字识别实验，从函数、矩阵和优化的角度理解神经网络，并研究 ReLU 与 Tanh 两种激活函数产生的内部表示。

**如果你没有人工智能基础，请先阅读：[从零理解神经网络与本项目：给数学系学生的逐步讲解](docs/从零理解神经网络与本项目.md)。**

这份讲解从图片如何变成向量开始，逐步解释神经网络、训练、反向传播、隐藏层激活和非负矩阵分解（NMF），包含手算例子、已有实验图、代码对照与结果解读。只需要线性代数、求导和一点概率基础。

## 项目要回答什么问题

识别手写数字需要先构造一个函数，把图片映射为 0 到 9 的类别。本项目进一步追问：这个函数的中间计算结果具有什么结构？能否用少量共同模式概括？

实验围绕三个问题展开：

1. ReLU 与 Tanh 的隐藏层激活，在非负性、零值比例和饱和程度上有什么差异？
2. 对非负激活矩阵做 NMF，需要多少成分才能较好地近似原矩阵？
3. 分解得到的成分是否集中在部分神经元上，是否与数字类别存在关联？

## 实验流程

```text
MNIST 图片与标签
    ↓
分别训练 ReLU 和 Tanh 分类网络
    ↓
提取全部测试图片的隐藏层激活
    ↓
统计激活分布、稀疏性、饱和与类别选择性
    ↓
对 ReLU 激活、Tanh 激活的正值部分分别做 NMF
    ↓
比较重构误差、成分稀疏度与类别关联
```

NMF 用于训练完成后的表示分析。网络训练和矩阵分解是两个独立的优化阶段。

## 数学对象与实验设置

每张 MNIST 图片为 $28\times28$ 的灰度图，展平后得到 $x\in\mathbb R^{784}$。网络计算为

$$
h=\sigma(Ux+b),\qquad s=Vh+c,
$$

其中 $h\in\mathbb R^{128}$ 是隐藏层激活，$s\in\mathbb R^{10}$ 是十个类别的分数，$\sigma$ 分别取 ReLU 或 Tanh。预测类别由最大分数的位置确定。

| 项目 | 当前设置 |
|---|---|
| 数据集 | MNIST：训练集 60,000 张，测试集 10,000 张 |
| 预处理 | `ToTensor()`，像素缩放到 $[0,1]$ |
| 网络结构 | 784 → 128 → ReLU / Tanh → 10 |
| 可训练参数 | 每个网络 101,770 个 |
| 训练 | Adam，学习率 0.001，批大小 128，5 轮 |
| 对照条件 | 相同初始参数、相同训练数据顺序，随机种子 42 |
| 激活矩阵 | 每种网络各一个 $10000\times128$ 矩阵 |
| NMF 成分数 | 4、8、12、16、24、32、48、64 |
| NMF 配置 | `nndsvda` 初始化、坐标下降、最多 500 次迭代 |

激活矩阵 $A$ 的每行对应一张图片，每列对应一个隐藏神经元。NMF 寻找非负因子

$$
A\approx WH,\qquad
W\in\mathbb R_{\geq0}^{10000\times k},\quad
H\in\mathbb R_{\geq0}^{k\times128}.
$$

$H$ 的每行描述一个神经元响应模式，$W$ 的每行描述一张图片使用各个模式的系数。这两个因子与网络权重 $U,V$ 是不同对象。

ReLU 激活天然非负，可以直接分解。Tanh 激活有正有负，当前实验先计算 $A^+=\max(A,0)$，再对正值部分做 NMF；这个处理会丢失负值信息。

## 已有实验结果

| 指标 | ReLU | Tanh / Tanh 正值部分 |
|---|---:|---:|
| 网络测试准确率 | 97.13% | 96.83%（原始 Tanh） |
| 原始激活的精确零值比例 | 38.43% | 0%（原始 Tanh） |
| 原始激活的负值比例 | 0% | 52.16%（原始 Tanh） |
| NMF 相对重构误差，$k=16$ | 0.3331 | 0.3833（Tanh 正值部分） |
| NMF 相对重构误差，$k=32$ | 0.2458 | 0.3236（Tanh 正值部分） |
| NMF 相对重构误差，$k=64$ | 0.1552 | 0.2426（Tanh 正值部分） |

分类结果见 [training_results.json](reports/metrics/training_results.json)，原始激活统计见 [activation_statistics.json](reports/activations/activation_statistics.json)，NMF 误差分别见 [ReLU 结果](reports/metrics/nmf_rank_comparison.json)与 [Tanh 正值部分结果](reports/metrics/tanh_positive_nmf_results.json)。

本次实验中，除 $k=4$ 外，ReLU 的相对重构误差均更低。这说明在当前设置下，它更容易被这些数量的非负成分近似。

解释结果时需要保留以下边界：

- 每种网络只训练一次，尚未进行多随机种子重复实验。
- Tanh 正值部分不同于完整的 Tanh 表示。
- 16 组 NMF 配置中有 7 组迭代次数达到上限，收敛情况仍需进一步检查。
- 重构误差、成分稀疏度与类别关联衡量不同性质；其中任一指标更好，都不足以单独证明表示更可解释。
- NMF 成分位于神经元空间，目前没有验证它们对应具体笔画。

## 怎样阅读这个仓库

| 内容 | 入口 |
|---|---|
| 从零理解概念与完整实验逻辑 | [中文逐步讲解](docs/从零理解神经网络与本项目.md) |
| 网络结构与中间激活的定义 | [models.py](src/mnist_nmf/models.py) |
| 网络训练与测试评估 | [train_models.py](scripts/train_models.py) |
| 隐藏层激活提取 | [extract_activations.py](scripts/extract_activations.py) |
| 神经元层面的统计分析 | [analyze_neuron_structure.py](scripts/analyze_neuron_structure.py) |
| ReLU 激活的 NMF | [run_nmf.py](scripts/run_nmf.py) |
| Tanh 正值部分的 NMF | [compare_relu_tanh_nmf.py](scripts/compare_relu_tanh_nmf.py) |
| 保存的图表与指标 | [reports/](reports/) |
| 正式实验报告与汇报幻灯片 | [最终汇报材料说明](reports/final/README.md) |

建议先读中文讲解第 1—5 节，理解网络与训练；再读第 6—12 节，理解激活矩阵与 NMF；最后读第 13—16 节，对照结果、脚本和术语。第 14、15 节列出了脚本的输入输出及当前实现细节。

## 环境

Python 3.10 或更新版本。依赖定义见 [pyproject.toml](pyproject.toml)，主要包括 PyTorch、Torchvision、NumPy、scikit-learn、Matplotlib、Pandas 和 PyYAML。

在已激活的 Python 环境中，从仓库根目录安装项目：

```bash
python -m pip install -e .
```

各实验脚本的执行依赖与读写位置见中文讲解。主要训练超参数目前直接写在脚本中，已保存结果对应当前这一次实验。
