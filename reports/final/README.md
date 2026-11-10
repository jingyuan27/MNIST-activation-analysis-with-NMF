# 最终汇报材料

本目录包含项目 **基于非负矩阵分解的神经网络隐藏层表示分析：ReLU 与 Tanh 在 MNIST 上的比较研究** 的两份正式材料：

- Beamer 汇报幻灯片（15 页，16:9）
- LaTeX 实验报告（ctexart）

两份文档中的数值都来自仓库里实际保存的实验结果：`reports/metrics/`、`reports/activations/`、`reports/nmf/`。

## 文件结构

```text
reports/final/
├── beamer/
│   ├── presentation.tex   # 幻灯片源文件
│   └── presentation.pdf   # 编译结果
├── report/
│   ├── report.tex         # 实验报告源文件
│   ├── references.bib     # BibTeX 参考文献（已联网核对）
│   └── report.pdf         # 编译结果
└── README.md
```

**关于图片：** 没有复制任何图片。两份文档都通过 `\graphicspath{{../../figures/}{../../figures/nmf/}}` 以相对路径直接引用 `reports/figures/` 中的原始 PNG，因此不需要单独的 `figures/` 目录。

**新增图表的绘制方式：**

- 分类准确率曲线、重构误差曲线和稀疏度柱状图，用 pgfplots 在 `.tex` 中直接绘制。
- 网络结构图和矩阵分解示意图，用 TikZ 绘制。
- 这些图表都不会生成新的图片文件。

## 编译环境

- TeX Live（已在 TeX Live 2026 上测试），需要 **XeLaTeX**
- `latexmk`
- **BibTeX**：仅报告需要，使用 `gbt7714` 宏包提供的 GB/T 7714 顺序编码制样式
- 需要的宏包：`ctex`、`pgfplots`、`tikz`、`booktabs`、`subcaption`、`gbt7714`、`hyperref`，TeX Live 完整安装已全部包含
- 中文字体：`ctex` 会按操作系统自动选择。Windows 上使用宋体、黑体等系统字体，无需额外配置

## 如何编译

编译命令需要在对应的子目录中执行，因为图片路径是相对于该目录的。

```bash
# 幻灯片
cd reports/final/beamer
latexmk -xelatex presentation.tex

# 实验报告（latexmk 会自动调用 bibtex 并多次编译，以解析引用）
cd reports/final/report
latexmk -xelatex report.tex
```

清理中间文件：

```bash
latexmk -c
```

不使用 latexmk 时，可以手动编译报告：

```bash
xelatex report.tex && bibtex report && xelatex report.tex && xelatex report.tex
```

## 修改作者、学校、日期

两份 `.tex` 文件开头都有以下三行，修改花括号中的内容即可：

```latex
\newcommand{\InfoAuthor}{作者姓名}
\newcommand{\InfoInstitute}{学校 / 学院}
\newcommand{\InfoDate}{汇报日期}   % 报告中为 {日期}
```

如需添加指导教师，可以在 `\author{...}` 中追加一行。

## 数据来源说明

- 分类性能来自 `reports/metrics/training_results.json`。
- 激活统计来自 `reports/activations/activation_statistics.json` 和 `reports/metrics/neuron_summary.json`。
- NMF 误差和迭代次数来自 `reports/metrics/nmf_rank_comparison.json` 和 `tanh_positive_nmf_results.json`。
- 以下数值仓库中没有对应的指标文件，是撰写文档时从已保存的 `.npz` 只读计算得到的，没有重新训练或重新分解：
  - 近零比例（|a|<0.1）
  - 平均 |a|
  - 原始激活的 Hoyer 稀疏度
  - 除 k=32 外，各秩下 H 的 Hoyer 稀疏度
  - 类别关联比值

  报告第 3.6 节和相关表注已经逐项标明这些数值的来源。
