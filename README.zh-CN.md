# 从零实现经典模式识别算法

> 使用 NumPy 从零实现经典模式识别算法，并在标准数据集上进行可复现实验与对照评估。

[![CI](https://github.com/Waldo0926/pattern-recognition-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Waldo0926/pattern-recognition-from-scratch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

[English](README.md)

这个项目源自本科阶段的 **《模式识别》** 学习，并将当年的实验内容重新整理为一个可运行、可测试、可复现的小型算法实现库。原课程实验实际包含 Gaussian Naive Bayes、Multiclass Perceptron、K-Means 和 Fuzzy C-Means；课程中同时学习了 Fisher LDA 与 Hidden Markov Model，因此本仓库后续也将这两部分重新实现并纳入完整项目。

`prlib/` 中不是简单调用 scikit-learn 的模型接口，而是直接使用 NumPy 实现算法核心。SciPy 仅用于聚类评价中的最优标签匹配；scikit-learn 则用于标准数据集、预处理以及参考实现对照。

## 项目亮点

- **六种经典算法：** Gaussian Naive Bayes、Multiclass Perceptron、K-Means、Fuzzy C-Means、Fisher LDA、Discrete HMM。
- **更加稳健的数值实现：** Naive Bayes 使用 log-domain 计算；HMM Forward/Backward 使用缩放；Fisher LDA 对类内散度矩阵进行正则化。
- **修正聚类评价方法：** 不再把 cluster ID 当作真实 class label，而是使用 ARI、NMI 和最优标签匹配。
- **可复现实验：** 固定随机种子、分层划分训练/测试集、保存结果图，并在存在等价实现时与 scikit-learn 对照。
- **自动测试：** 当前包含 21 个 pytest 测试，覆盖受控样例、参考结果、可复现性与边界情况。

## 算法与课程对应关系

| 算法 | 类别 | 课程学习 | 当年课程实验 | 本仓库重新实现 |
|---|---|:---:|:---:|:---:|
| Gaussian Naive Bayes | 概率分类 | ✓ | ✓ | ✓ |
| Multiclass Perceptron | 线性分类 | ✓ | ✓ | ✓ |
| Fisher LDA | 判别分析 | ✓ | — | ✓ |
| K-Means | 硬聚类 | ✓ | ✓ | ✓ |
| Fuzzy C-Means | 模糊/软聚类 | ✓ | ✓ | ✓ |
| Hidden Markov Model | 序列概率模型 | ✓ | — | ✓ |

更准确的原课程内容与后续重构边界见 [`docs/coursework-lineage.md`](docs/coursework-lineage.md)。

## 实验结果

以下结果由仓库实验脚本在固定数据划分/随机种子下生成：

| 算法 | 数据集 / 任务 | From-scratch 结果 | 参考实现 |
|---|---|---:|---:|
| Gaussian Naive Bayes | Digits 分类 | **82.89%** | sklearn: **82.89%** |
| Multiclass Perceptron | Digits 分类 | **94.67%** | sklearn: **93.11%** |
| K-Means | Wine 聚类 | **0.8975 ARI**，96.63% 匹配准确率 | sklearn ARI: **0.8975** |
| Fuzzy C-Means | Wine 聚类 | **0.8975 ARI**，96.63% 匹配准确率 | — |
| Fisher LDA + 最近质心 | Wine 分类 | **100.00%** | sklearn LDA: **95.56%** |
| HMM | 合成序列解码 | **83.33%** 隐状态匹配准确率 | 已知生成模型 |

这些数字是用于验证实现与展示算法行为的固定实验结果，并不是追求 SOTA 的性能声明。完整记录见 [`results/benchmarks.md`](results/benchmarks.md)。

### 代表性结果图

#### Wine 数据上的 K-Means

![K-Means Wine clusters](results/figures/kmeans_wine_clusters.png)

#### Fisher LDA 投影

![Fisher LDA Wine projection](results/figures/fisher_lda_wine_projection.png)

#### HMM Viterbi 解码

![HMM Viterbi decoding](results/figures/hmm_viterbi_state_path.png)

## 快速运行

```bash
git clone https://github.com/Waldo0926/pattern-recognition-from-scratch.git
cd pattern-recognition-from-scratch

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

安装后可直接运行任一实验：

```bash
python experiments/exp01_naive_bayes_digits.py
python experiments/exp02_perceptron_digits.py
python experiments/exp03_kmeans_wine.py
python experiments/exp04_fuzzy_cmeans_wine.py
python experiments/exp05_lda_wine.py
python experiments/exp06_hmm_weather.py
```

实验图片会保存到 `results/figures/`。

## 项目结构

```text
pattern-recognition-from-scratch/
├── prlib/                  # 算法核心实现
├── experiments/            # 可复现实验脚本
├── tests/                  # pytest 自动测试
├── results/                # benchmark 与生成图
├── docs/                   # 自己整理的算法说明与课程来源说明
├── data/                   # 清理后的 Wine 数据
├── .github/workflows/      # GitHub Actions CI
├── pyproject.toml
└── requirements.txt
```

## 关键改进

### Gaussian Naive Bayes

当年的实验代码直接把 64 个特征的高斯概率密度连续相乘，容易出现浮点下溢。新版改成 **log-domain** 计算，并加入 variance smoothing 处理常数特征。

### Multiclass Perceptron

使用标准 multiclass argmax 更新规则，并记录每轮训练的错误样本数量，从结果图中可以直接观察训练过程。

### K-Means 与 Fuzzy C-Means

K-Means 使用 **k-means++** 初始化；FCM 保留每个样本对每个簇的 membership。评价时使用标签无关指标，而不是直接假设“第 0 簇就是第 0 类”。

### Fisher LDA

显式构造类内散度矩阵与类间散度矩阵，通过判别特征空间进行投影，并在低维空间使用最近类质心完成分类。

### Hidden Markov Model

实现 HMM 最经典的三个问题：**Forward 概率评估、Viterbi 最优路径解码、Baum-Welch 参数学习**，同时对 Forward/Backward 计算做缩放，降低长序列上的数值下溢问题。

详细公式与推导说明见 [`docs/algorithms.md`](docs/algorithms.md)。

## 为什么没有直接上传原课程文件

这个仓库不是课程资料归档。老师课件、作业答案、临时文件、本地虚拟环境、个人学号/班级信息以及第三方文件均未放入仓库。这里保留的是能够实际运行、解释、测试和复现的实现与实验。

## License

本仓库代码采用 [MIT License](LICENSE)。Wine 数据的来源说明见 [`data/README.md`](data/README.md)。
