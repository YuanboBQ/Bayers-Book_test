# <center> Lecture8 : A Simple Linear Regression Model with PyMC </center>  

## <center> Instructor: Dr. Hu Chuan-Peng </center>

## 序言  

> ⭐ 在之前的课程中，我们学习了贝叶斯统计方法和使用代码实现统计模型。  
> 
> 然而，我们并没有使用贝叶斯模型深入到真正的心理学研究当中。  
> 
> 从本节课开始，我们将通过一个个真实的案例，带领大家进入心理学贝叶斯统计分析的世界。
>
本节课我们将使用一个简单的线性回归模型，通过PyMC来建立这个模型，并走完贝叶斯分析的大部分流程。后续，我们将把这个贝叶斯模型的分解式和框架应用到更复杂的线性模型上，直至层级模型。
💡希望大家开始投入更多时间进行**练习**，尽量搞懂课堂相关的练习部分。大作业也可以开始提前准备。因为这节课并没有涉及太多新的统计知识，包括模型，我默认大家已经有一定的了解。我们主要是在贝叶斯框架下，换一个视角来做这些熟悉的统计方法。今天，我们就从一个非常简单的线性回归模型开始。

## 研究示例： 自我加工优势 (Self-prioritization Effect, SPE)  

在本节课，我们关注的研究问题是 “自我作为人类独特的心理概念，是否在也会促进认知加工的表现？”。  

特别地，我们关注的是自我在知觉匹配任务中的作用，**探究自我和他人条件下，人们的认知加工差异，尤其是在反应时间上的表现。**  

2012年，Jie Sui开发了一个新的范式，即在实验室中让被试将原本中性的刺激与自我或他人关联，然后进行认知任务。例如，我们让被试学习三角形代表他自己，圆形代表另一个人，然后进行匹配任务。任务是判断呈现的图形和标签是否匹配，即是否符合他们刚刚学到的关系。当然，这个关系在被试之间是平衡的。
通过这个实验，我们可以在一定程度上消除自我熟悉性的影响。因为在日常生活中，我们对自己相关信息肯定更加熟悉。所以当我们在实验室中及时习得这种自我相关性时，相对来说是更好的数据控制。
通过这个实验，我们基本上能够发现，当一个原本中性的刺激与自我关联后，被试在进行匹配任务时会有更快更准的反应。

> 探究自我加工的优势通畅使用匹配范式任务中“自我（self）”和“他人（other）”的认知匹配之间的关系 (Sui et al., 2012)。  
>
> * 在自我匹配任务中，被试首先学习几何图形和身份标签的关系。例如，三角形代表自我；圆形代表他人。在随后的知觉匹配判断任务中，需要判断所呈现的几何图形和文字标签是否与之前学习的关系相匹配。  
> * 想象一下你在完成自我匹配任务时，面对不同的刺激：有时你可能会觉得某个“自我”相关的图像比“他人”相关的更具吸引力，或许这反映了你对自己本身具有更多的关注。  

![Image 1](./img%201.png)  

> Sui, J., He, X., & Humphreys, G. W. (2012). Perceptual effects of social salience: Evidence from self-prioritization effects on perceptual matching. Journal of Experimental Psychology: Human Perception and Performance, 38(5), 1105–1117. <https://doi.org/10.1037/a0029792>  

根据Sui et al., （2012）的文章，我们假设，**在“自我”条件下，个体的反应时间会快于在“他人”条件下的反应时间。** 那么在贝叶斯的框架下，我们应该如何解决我们的研究问题以及验证我们的研究假设是否正确呢？  

![Image 2](./img%202.png)  

在本节课中，我们将学习如何使用大家所熟悉的简单线性模型 (linear regression model) 来检验心理学效应。  

包括以下内容：  

1. **简单线性模型**。  
2. **先验预测检验**。  
3. 模型拟合和诊断。  
4. 后验推理。  
5. **模型检验 (后验预测检验)**。  

在**传统统计学**中，研究问题通常转化为**假设检验**问题。配对样本t检验是解决此类问题的一个常用方法，但它实际上是回归分析的一个特例。本节内容将阐述**从研究问题的明确到数据收集后，如何选择合适的统计模型，以及如何构建似然函数**。在**贝叶斯分析**中，除了似然函数，还需考虑先验分布。
**实验预测检验**是实验设计后验证模型适宜性的关键步骤。若模型适宜，将采用MCMC方法进行推断，以获得后验分布。随后，评估MCMC结果的合理性，并进一步评估模型的预测效果，以确保模型的可用性。本节将通过一个简单的线性回归模型，演示从模型构建到基于后验分布的统计推断的完整流程，并展示代码实现方法。这包括线性模型的构建、先验预测检验、模型拟合与诊断、后验检验等步骤，旨在确保推断的严谨性。
<div style="padding-bottom: 20px;"></div>
我们使用的数据来自于Kolvoort等人（2020），该数据集包含了多个被试在自我匹配范式下的行为数据。数据集涉及了不同年龄、性别、文化背景的健康成年被试。  

数据可在和鲸平台及Gitee平台上获取。在Python环境中，利用pandas库读取数据。本研究关注的自变量是标签（label），用以区分实验条件是自我相关还是他人相关。因变量为反应时间（RT），单位为秒。每条数据记录代表一个试验（trial）。在认知实验中，每位受试者在每个条件下需完成多次试验，可能是50次、60次甚至超过100次。每行数据代表单个试验中的一个观测值。

* 我们使用 `pandas` 的 `read_csv` 方法来读取数据 `Kolvoort_2020_HBM_Exp1_Clean.csv` (数据已经预先存放在和鲸平台中)。  
* 数据包含多个变量，选择我们需要的`Label` 表示标签（self / other），`RT_sec` 表示被试的反应时间。  
* 每一行(index)表示一个trial数。  

> * 数据来源: Kolvoort, I. R., Wainio‐Theberge, S., Wolff, A., & Northoff, G. (2020). Temporal integration as “common currency” of brain and self‐scale‐free activity in resting‐state EEG correlates with temporal delay effects on self‐relatedness. Human brain mapping, 41(15), 4355-4374.  

```python

# 导入 pymc 模型包，和 arviz 等分析工具 
import pymc as pm
import arviz as az
import seaborn as sns
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

# 忽略不必要的警告
import warnings
warnings.filterwarnings("ignore")
# 通过 pd.read_csv 加载数据 Kolvoort_2020_HBM_Exp1_Clean.csv
try:
  df_raw = pd.read_csv('/home/mw/input/bayes3797/Kolvoort_2020_HBM_Exp1_Clean.csv')
except:
  df_raw = pd.read_csv('2024/data/Kolvoort_2020_HBM_Exp1_Clean.csv')

df_raw.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Subject</th>
      <th>Age</th>
      <th>Handedness</th>
      <th>First_Language</th>
      <th>Education</th>
      <th>Countryself</th>
      <th>Countryparents</th>
      <th>Shape</th>
      <th>Label</th>
      <th>Matching</th>
      <th>Response</th>
      <th>RT_ms</th>
      <th>RT_sec</th>
      <th>ACC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>201</td>
      <td>18</td>
      <td>r</td>
      <td>English/Farsi</td>
      <td>High School</td>
      <td>Iran/Canada</td>
      <td>Iran</td>
      <td>3</td>
      <td>2</td>
      <td>Matching</td>
      <td>1</td>
      <td>753</td>
      <td>0.753</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>201</td>
      <td>18</td>
      <td>r</td>
      <td>English/Farsi</td>
      <td>High School</td>
      <td>Iran/Canada</td>
      <td>Iran</td>
      <td>3</td>
      <td>2</td>
      <td>Matching</td>
      <td>1</td>
      <td>818</td>
      <td>0.818</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>201</td>
      <td>18</td>
      <td>r</td>
      <td>English/Farsi</td>
      <td>High School</td>
      <td>Iran/Canada</td>
      <td>Iran</td>
      <td>1</td>
      <td>3</td>
      <td>Matching</td>
      <td>1</td>
      <td>917</td>
      <td>0.917</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>201</td>
      <td>18</td>
      <td>r</td>
      <td>English/Farsi</td>
      <td>High School</td>
      <td>Iran/Canada</td>
      <td>Iran</td>
      <td>3</td>
      <td>2</td>
      <td>Matching</td>
      <td>1</td>
      <td>717</td>
      <td>0.717</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>201</td>
      <td>18</td>
      <td>r</td>
      <td>English/Farsi</td>
      <td>High School</td>
      <td>Iran/Canada</td>
      <td>Iran</td>
      <td>3</td>
      <td>2</td>
      <td>Matching</td>
      <td>1</td>
      <td>988</td>
      <td>0.988</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

在Jupyter Notebook中进行数据分析的初步步骤包括挂载所需数据并读取。通过查看数据的前五行，我们可以初步了解数据结构，包括被试的标签和年龄等信息，尽管这些信息在当前分析中并非关注重点。我们的核心关注点是反应时间（RT）及其毫秒值，以及准确率，尽管现阶段我们主要关注RT和与之相关的标签（label）。
为了简化演示过程，我们将选取单个被试在特定条件下（例如匹配条件）的数据。这样做的目的是为了使模型尽可能简化，便于理解和演示。接下来，我们将对label的含义进行详细说明，并展示如何从简化的数据集中提取和分析关键信息：

```python
# 筛选出被试"201"，匹配类型为"Matching"的数据
df_raw["Subject"] = df_raw["Subject"].astype(str)
df = df_raw[(df_raw["Subject"] == "201") & (df_raw["Matching"] == "Matching")]

# 选择需要的两列
df = df[["Label", "RT_sec"]]

# 重新编码标签（Label）
df["Label"] = df["Label"].map({1: 0, 2: 1, 3: 1})

# #设置索引
df["index"] = range(len(df))
df = df.set_index("index")

# 显示部分数据
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>RT_sec</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.753</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.988</td>
    </tr>
  </tbody>
</table>
</div>

在最终的模型构建中，我们将关注两个核心变量：标签（label）和反应时间（second），即以秒为单位的RT。基于这两个变量，我们将建立一个简洁的回归模型。
在正式进行模型构建之前，我们建议先进行**数据的可视化**，这有助于直观理解数据分布和潜在的模式，为后续的模型分析提供直观的依据。
进一步可视化数据情况：  

* 我们使用 `seaborn` 的 `boxplot` 方法来进行可视化。  
* 其中横轴为Label `x="Label"`，纵轴为反应时间 `y="RT_sec"`。  
  
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 计算每个Label条件下的均值
mean_values = df.groupby('Label')['RT_sec'].mean()

# 使用 seaborn 绘制箱线图，展示 Matching 条件下反应时间的分布情况
plt.figure(figsize=(5, 3.2))
sns.boxplot(x="Label", y="RT_sec", data=df)

plt.plot(mean_values.index, mean_values.values, marker='o', color='r', linestyle='-', linewidth=2)

# 移除上面和右边的边框
sns.despine()

# 添加标签
plt.xlabel("Label Condition (0=self, 1=other)")
plt.ylabel("Reaction Time (sec)")
plt.show()

```

![alt text](./img%203.png)

>💡**强烈推荐大家进行数据可视化：**
>
> * 可视化允许我们直接观察原始数据，而不是立即进行统计分析。直接跳过数据探索阶段而进入分析是不恰当的，因为即使统计结果相似，数据的实际模式也可能截然不同。例如，即使相关系数保持不变，数据的图形表示可能会有显著差异，这强调了可视化的重要性，以确保所选统计模型的合理性。
> * 对于特定的数据集，可视化揭示了常见的反应时间特征。在实验室条件下，被试的反应时间通常在一秒以内，大约在0.6到0.7秒之间，并且存在超过一秒的分布。此外，数据还显示出自我优势效应的趋势，即在自我条件下的反应时间相对较短。这种可视化使我们能够识别数据中潜在的趋势。
> 然而，仅仅识别趋势是不够的。当我们绘制所有单个试验的数据时，会发现数据之间存在重叠，这使得我们无法立即确定一个条件是否总是优于另一个条件。这种数据重叠强调了进行严格统计检验的必要性，以确定不同条件之间是否存在显著差异。统计检验在这里至关重要，因为它帮助我们超越直观观察，量化参数之间的差异，从而做出更准确的推断。
>
> 在数据分析的过程中，我们经常会遇到一个误区，即认为只有通过统计分析才能做出推断。然而，实际情况并非总是如此。在某些情况下，纯描述性的数据已经足够明显，以至于我们可以直接从中得出结论，无需进行复杂的统计分析。当数据差异显著到肉眼即可辨识时，统计分析可能变得多余。
> 统计分析的真正价值在于，当我们的直观观察无法达成共识，或者需要更深入地了解数据背后的参数差异时，它提供了一种量化的方法。在这种情况下，统计分析成为揭示数据背后差异的重要工具。此外，统计结果的应用通常有两个目的：支持理论发展或辅助决策制定。
> 在实际应用场景中，例如心理健康领域，有时简单的观察就足以判断干预措施的效果，无需进行AB test或统计分析。当决策目标明确，且数据变化明显时，原始数据本身就可能提供足够的信息，使得统计判断变得不必要。
> 💡因此，这里有两个重要的注意事项需要强调：
>
> * 在进行数据分析之前，一定要看**原始数据**。不应盲目地应用标准流程，而忽视了对数据本身的理解，这种做法是机械且缺乏洞察力的。
> * 认识到统计分析**并非在所有情况下都是必需的**。在许多情况下，我们可以直接从数据中推断出结论，而无需依赖统计方法。

## 简单线性回归：使用线性模型表示两种条件下反应的差异  

### 频率学派视角下的回归模型  

让我们先回顾在传统频率学派的视角下，回归模型的建立和检验一般基于参数估计和假设检验：  

1. **构建模型**：我们可以使用一个简单线性回归模型，其中反应时间（RT_sec）作为因变量，自变量为二分离散变量（Label）。我们可以将 `self` 编码为 0，`other` 编码为 1，这样模型将估计出“自我”条件相较于“他人”条件在反应时间上的效应。  

模型形式为：  

$$  
   RT_{sec} = \beta_0 + \beta_1 \cdot Label + \epsilon  
$$  

其中，$\beta_0$ 表示“self”条件下的平均反应时间，$\beta_1$ 表示other条件下相较于self条件的反应时间差异。  

2. **假设检验**：在该模型中，$\beta_1$ 的显著性可以用 t 检验来判断。如果 $\beta_1$ 显著不为 0（例如 $p < 0.05$），则说明自我条件下的反应时间显著不同于他人条件，即存在自我加工的优势。  

3. **模型解释**：若 $\beta_1$ 为负值，则表明自我条件的反应时间较短，暗示自我加工速度较快。

### 贝叶斯视角下的回归模型  

在贝叶斯视角下，回归模型的建立和检验不同于传统的假设检验，而是通过对参数的后验分布进行推断：  

1. **构建贝叶斯模型**：贝叶斯模型和频率学派的回归模型形式相同，但其参数估计基于贝叶斯推断。我们会为 $\beta_0$ 和 $\beta_1$ 指定先验分布（例如，正态分布），并结合观测数据计算其后验分布：  

$$  
RT_{sec} \sim \mathcal{N}(\beta_0 + \beta_1 \cdot Label, \sigma^2)  
$$  

1. **计算后验分布**：使用贝叶斯推断方法（如 MCMC 采样）得到 $\beta_1$ 的后验分布。  

2. **显著性检验**：通过后验分布检验 $\beta_1$ 是否显著，例如计算 $\beta_1 > 0$ 或 $\beta_1 < 0$ 的概率，或计算最高密度区间（HDI）。如果 95% HDI 不包含 0，可以认为自我条件和他人条件在反应时间上的差异是显著的。  

3. **模型解释**：在贝叶斯框架下，我们不仅可以观察参数的点估计（如 $\beta_1$ 的均值），还可以通过后验分布和 HDI 提供更加直观的置信水平解释。

在前面，我们讨论了频率主义（经典统计）视角下的回归模型。在贝叶斯回归中，基本思路一致，但表达形式有所不同。在频率主义框架下，模型包含一个误差项，用于解释数据的变异性。
回归模型的核心任务是对数据的均值进行预测。当$X$取某一具体值时，我们假定目标变量$Y$来源于某个正态分布。该分布的均值由线性模型确定，即通过自变量的线性组合计算得到；而贝叶斯框架仅是换一种方式描述相同的问题。需要注意的是，传统线性回归假设残差（误差项）服从正态分布，这是理论成立的重要前提条件。这一假设确保通过线性组合预测的均值具有合理性。



**⭐贝叶斯回归模型的可视化表达**  

* <span style = "color: orange;">预测值</span> $\mu_i$(即直线上橙色的点)可以写为：$\mu_i = \beta_0 + \beta_1 X_i(1)$  

* 从图上可以看到<span style = "color: orange;">预测值</span>和<span style = "color: gray;">实际值</span> (即灰色的散点)之间存在出入，实际值会在预测值附近波动  

* 那么实际值可以看作服从以$\mu_i$为均值，标准差为$\sigma$的正态分布，即：$Y_i \sim N(\mu_i, \sigma ^ 2)$  

![Image 5](./img%205.png)  

*(改编自：https://saylordotorg.github.io/text_introductory-statistics/s14-03-modelling-linear-relationships.html)*  

<br>

对于$X_0$和$X_1$，我们预测的是什么？实际上是预测在某个特定条件下$Y$的均值。当$X=1$时，我们预测的是$Y$的均值。以这个均值为中心，数据呈正态分布，并在均值上下波动。
在这种情况下，我们可以预测不同条件下的均值。例如，假设有两个条件$i=1$和$i=2$，分别对应两个均值$μ_1$和$μ_2$。每个均值都对应一个正态分布，数据围绕其均值波动。

回到[贝叶斯视角下的回归模型](#贝叶斯视角下的回归模型)，我们实际上是将传统建模方式应用到贝叶斯框架中，构建一个简单的回归模型，即基于正态分布的回归模型。这被称为简单回归模型，因为它仅涉及一个正态分布，用以描述因变量的分布特性。
在贝叶斯框架下构建回归模型的流程如下：
>根据数据特点（包括因变量和自变量），结合经验判断，选择适合的模型。
>对模型参数设置先验分布和似然函数。
>利用概率编程语言计算后验分布。


通过后验分布，我们可以进行统计推断。虽然贝叶斯方法中没有传统的“显著性检验”，但我们可以使用后验分布的最高密度区间（HDI）和区域等同于零（ROPE）来推断参数是否显著：
>- 如果后验分布与 ROPE 完全不重叠，则可认为参数显著（不等于 0）。
>- 类似于传统统计中对显著性的解释。

此外，还可以根据后验分布计算效应量，或利用贝叶斯因子进行统计推断。关于贝叶斯因子的计算，我们将在下一节详细讲解，补充 HDI 之外的推断方法。

<div style="padding-bottom: 20px;"></div>

**贝叶斯回归模型的数学表达式**  

$$  
\begin{align*}  
\beta_0   &\sim N\left(m_0, s_0^2 \right)  \\  
\beta_1   &\sim N\left(m_1, s_1^2 \right)  \\  
\sigma    &\sim \text{Exp}(\lambda)        \\  
&\Downarrow \\  
\mu_i &= \beta_0 + \beta_1 X_i      \\  
&\Downarrow \\  
Y_i | \beta_0, \beta_1, \sigma &\stackrel{ind}{\sim} N\left(\mu_i, \sigma^2\right). \\  
\end{align*}  
$$  

* 回归模型需满足如下假设：  

    1. 独立观测假设:每个观测值$Y_i$是相互独立的，即一个被试的反应时间不受其他被试的影响。  

    2. 线性关系假设: 预测值$\mu_i$和自变量$X_i$之间可以用线性关系来描述，即：$\mu_i = \beta_0 + \beta_1 X_i$  

    3. 方差同质性假设： 在任意自变量的取值下，观测值$Y_i$都会以$\mu_i$为中心，同样的标准差$\sigma$呈正态分布变化（$\sigma$ is consistent）  

**表达式讲解：**
在这个回归模型中，正态分布有三个参数：$β_0$、$β_1$和$σ$。每一个观测值$Y_i$都是从其对应的正态分布中采样而来，这里的$i$对应不同的$X$取值。在我们的模型中，只有两个条件$X_0$和$X_1$，每个条件对应一个正态分布。由于有三个参数，因此在进行贝叶斯分析时，需要为这些参数指定先验分布。一旦确定了先验分布和似然函数（模型的定义），我们就可以进行后续的贝叶斯推断。

回归模型的假设与传统模型类似。首先，假设观测值$Y_i$是独立的，即每个观测值都不受其他观测值的影响。其次，模型假定误差项服从正态分布，并且遵循方差同质性假设，即所有条件下的$σ$相同（没有下角标标记）。

然而，贝叶斯模型的**灵活性**使得我们可以进一步扩展这个假设。例如，我们可以为每个条件的方差$σ$添加下角标，以允许其在不同条件下变化。这种灵活性正是贝叶斯模型的一个显著特点，允许我们在建模过程中根据数据的特点进行更为精细的调整。

## 定义先验  

在贝叶斯的分析框架中，我们需要为模型中的每个参数设置先验分布。  

而根据之前的模型公式(数据模型)可发现，我们的$Y$为被试反应时间(RT_sec)，$X$为标签（Label），并且存在三个未知的参数$\beta_0，\beta_1，\sigma$ 。  

因此，我们需要对每个未知的参数定义先验分布。  

$$  
\beta_0    \sim N\left(m_0, s_0^2 \right)  \\  
\beta_1   \sim N\left(m_1, s_1^2 \right) \\  
\sigma \sim \text{Exp}(\lambda)  
$$  

接下来，我们逐步完成模型的各个部分。首先，我们要为每个参数指定先验分布。因变量是反应时自变量仅包含两个条件$X_0$和$X_1$，因此我们可以从一个基本的模型形式开始。例如，我们可以假定$β_0$和$β_1$都服从正态分布，这是一个合理的假设，可以作为模型的起点。在这个假设下，$β_0$和$β_1$各自有均值和标准差，表示它们的分布特性。
对于方差$σ$，由于方差不可能为负值，因此我们需要确保先验分布的选择使得$σ$始终为正值。在实践中，方差的先验分布可以有多种选择，具体取决于建模者的偏好和经验。常见的选择之一是使用指数分布，因为它能确保值为正。此外，也可以选择半正态分布（half Normal），即正态分布的正半部分，这也是一种可行的选择。其他的分布类型，只要满足方差为正的条件，也都可以使用。虽然不同领域和研究者的选择可能有所不同，但在心理学研究中，指数分布（Exponential Distribution）通常被认为是一个较为理想的选择。尽管如此，选择哪种分布以及如何确定其参数仍需根据实际数据和研究背景来进一步调整。

> * 参数的前提假设(assumptions):  
>   * $\beta_0，\beta_1，\sigma$ 之间相互独立  
> * 此外，规定 $\sigma$ 服从指数分布，以限定其值恒为正数  
> * 其中，$m_0，s_0，m_1，s_1$为超参数  
>   * 我们需要根据我们对$\beta_0$和$\beta_1$的先验理解来选择超参数的范围  
>   * 比如，$\beta_1$反映了标签从 self 切换到 other 时，反应时间的平均变化值；$\beta_0$反映了在 other 条件下的基础反应时间  

在贝叶斯建模中，确定先验分布常常是最具挑战性的部分。我们已经有了模型和数据，但要进行贝叶斯分析，必须首先确定先验分布的具体形式和取值，才能执行后续的MCMC（马尔可夫链蒙特卡洛）计算。因此，首要任务是能够让模型成功运行。

为确定先验分布，我们可以根据数据的特征和每个参数的实际意义来大致推测其取值。比如，$\beta_0$代表自变量$X=0$时某一条件下的反应时间（RT）。我们可以将其看作反映人类在实验中的反应时间范围。如果以秒为单位，显然反应时间不可能是千秒、万秒或十万秒这样的量级。基于此，我们可以推测出$\beta_0$的一个合理范围。
类似地，$\beta_1$反映的是两种条件下反应时间的差异。通常来说，$\beta_1$的量级应该比$\beta_0$小，因为它表示的是两条件间的差异。若差异过大，则可能说明实验设计或数据本身存在问题。因此，根据这些基本原则，我们可以确定一个大致的先验范围，从而为后续的贝叶斯分析提供合理的起点。

**指定超参数**  

$$  
\begin{equation}  
\begin{array}{lcrl}  
\text{data:} & \hspace{.05in} &   Y_i | \beta_0, \beta_1, \sigma & \stackrel{ind}{\sim} N\left(\mu_i, \sigma^2\right) \;\; \text{ with } \;\; \mu_i = \beta_0 + \beta_1X_i \\  
\text{priors:} & & \beta_{0}  & \sim N\left(5, 2^2 \right)  \\  
                    & & \beta_1  & \sim N\left(0, 1^2 \right) \\  
                    & & \sigma   & \sim \text{Exp}(0.3)  \\  
\end{array}  
\end{equation}  
$$  

这里，我们根据生活经验或直觉对超参数进行了定义：  

* 其次，我们假设 $\beta_0$ 服从均值为 5，标准差为 2 的正态分布,代表：  
  * 当实验条件为 self（编码为 0）时，反应时间的平均值大约为 5 秒。  
  * 截距值可能在 3 ± 7 秒 的范围内波动，反映了在 self 条件下的反应时间预估  
  
* 我们假设 $\beta_1$ 服从均值为 0，标准差为 1 的正态分布，代表：  

  * (斜率)将其均值指定为 1，表示我们预期在 self 和 other 条件下的反应时间差异较小。  

    * 这个影响的量是变化的，范围大概在 -1  ± 1。  

* 最后，我们假设 $\sigma$ 服从指数分布，其参数为0.3。  

  * 参数0.3 意味着标准差通常集中在较小的正数范围内，使反应时间在预测值$\mu_i$附近波动。  
  * 这一设置允许较小到中等的波动，但大部分数据应集中在 0 到 10 秒的合理反应时间范围内。  

<div style="padding-bottom: 20px;"></div>

我们在确定贝叶斯回归模型中的先验分布时，首先依据对人类反应时间的基本认识。具体来说，$\beta_0$代表某一条件下的基础反应时间，可以理解为所有人在认知实验中的反应时间范围。因此，我们可以将其先验分布设置为一个正态分布，例如均值为5秒，标准差为2秒。

对于$\beta_1$，它表示两个条件之间的反应时间差异。基于常识，我们可以将其先验分布设置为均值为0，标准差为1的正态分布，假设差异值可能是正的或负的，通常不会超过一秒。

对于方差$σ$，我们假定其反映了反应时间的离散程度。通常情况下，人的反应时间标准差较小，因此可以选择一个集中于较小值的先验分布，例如指数分布，允许方差在一定范围内波动，但仍保持合理性。

在本节课练习和实际应用中，大家可以根据自己的需求调整贝叶斯模型中的先验分布。先验的设置通常基于对数据的直觉和常识，但如何判断先验是否合理呢？关键是要确保它不违背常识。例如，我们可以通过数据的范围来构建先验，确保先验与数据一致。然而，一般来说，不推荐设定一个过于严格或狭窄的先验范围。

当先验过于强时，意味着它与数据不符，这时需要大量的数据才能更新先验至后验。相反，如果先验适度且较为宽松（weakly informed prior），即使数据量不大，也能有效更新至后验，这样可以避免先验对推断过程的过度影响。

举个例子，假设我们预期反应时间大约在1至2秒之间，设定先验为均值1秒，标准差2秒。这虽然可以，但标准差为2意味着反应时间可能会出现负值，这是不合常理的。因此，我们需要根据常识来调整先验的范围，避免不合理的结果。

此外，在某些情况下，我们也可以考虑不使用正态分布，而是根据数据的实际形态选择合适的分布类型。这些方法会在后续的课程中进一步讲解，尤其是当我们需要使用广义线性模型时。

可视化指定超参下的先验：  
```python
# 导入库

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# 定义新的先验分布的参数

mu_beta0 = 5
sigma_beta0 = 2
mu_beta1 = 0
sigma_beta1 = 1
lambda_sigma = 0.3

# 生成 beta_0 的先验分布值

x_beta0 = np.linspace(-5, 15, 1000)  
y_beta0 = stats.norm.pdf(x_beta0, mu_beta0, sigma_beta0)

# 生成 beta_1 的先验分布值

x_beta1 = np.linspace(-5, 5, 1000)
y_beta1 = stats.norm.pdf(x_beta1, mu_beta1, sigma_beta1)

# 生成 sigma 的先验分布值

x_sigma = np.linspace(0, 10, 1000)  
y_sigma = stats.expon.pdf(x_sigma, scale=1/lambda_sigma)

# 绘制先验分布图

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 绘制 beta_0 的先验分布

axes[0].plot(x_beta0, y_beta0, 'k-')
axes[0].set_title(r"$N(5, 2^2)$")
axes[0].set_xlabel(r"$\beta_0$")
axes[0].set_ylabel("pdf")

# 绘制 beta_1 的先验分布

axes[1].plot(x_beta1, y_beta1, 'k-')
axes[1].set_title(r"$N(0, 1^2)$")
axes[1].set_xlabel(r"$\beta_1$")
axes[1].set_ylabel("pdf")

# 绘制 sigma 的先验分布

axes[2].plot(x_sigma, y_sigma, 'k-')
axes[2].set_title(r"Exp(0.3)")
axes[2].set_xlabel(r"$\sigma$")
axes[2].set_ylabel("pdf")

# 调整布局并显示图表

sns.despine()
plt.tight_layout()
plt.show()
```
![img 4](./img%204.png)

我们从先验的可视化开始，帮助大家更直观地理解各个参数的分布。我们当前的模型包括三个参数：$\beta_0$、$\beta_1$和 $\sigma$。通过可视化这些参数的分布，我们可以更好地建立直觉理解，尤其对于统计学背景的人来说，这种直观的感受非常重要。通过可视化，我们能够直观地看到这些参数的可能取值范围，从而增强我们对先验分布的认识。

例如，当我们使用$\beta_0= 5$ 和$\beta_1= 2$ 这两个参数时，虽然理论上β1的值可以是正负的，但可视化显示在正态分布下，我们可能会看到少量负值。这是因为$\beta_1$代表的是条件间的差异，因此其正负取值是合理的。至于$\sigma$，它是标准差，理论上应为正值，而通过可视化，我们可以发现其大部分取值集中在较小的范围内，如8或10以内，这是一个合理且可接受的范围。


### 先验预测检验(prior predictive check)  

🤔有些同学可能认为这个先验的定义过于随意，甚至有些不靠谱。 那我们是否可以检验先验的合理性，以及适当的调整这个先验呐？  

**我们通过代码来说明，如何进行先验预测检验**  

首先根据公式，先验模型为：  

$$  
\begin{align*}  
\text{priors:} & & \beta_{0}  & \sim N\left(5, 2^2 \right)  \\  
                    & & \beta_1  & \sim N\left(0, 1^2 \right) \\  
                    & & \sigma   & \sim \text{Exp}(0.3)  \\  
\end{align*}  
$$

为了检验先验分布的合理性，我们可以使用一种方法叫做先验预测检验。这一方法的核心思想是，从先验分布中随机抽取一组参数，并将这些参数带入我们的数据模型中，生成预测值。通过这种方式，我们能够判断这些预测值是否符合我们基于经验的常识。

具体来说，先验分布本质上是从一个概率分布中抽取的，因此我们可以在这个分布下随机抽取多个样本。每个样本代表了我们在没有看到数据之前对模型的信念。然后，使用这些抽取的样本参数，我们将其带入模型，生成一组预测数据，进而检查这些预测值是否符合实际的常识性预期。这个过程实际上是一种蒙特卡洛采样方法，通过随机抽取大量的样本来估算数据的分布，帮助我们验证先验设定的合理性。

**先验预测检验的大致思路**  

1. 在先验中随机抽取200组$\beta_0, \beta_1$值  
2. 生成假数据自变量X  
3. 生成200条 $\beta_0 + \beta_1 X$ 生成预测的反应时间数据  
4. 观察生成的反应时间数据是否在合理范围内，评估先验假设的合适性  

1. 在先验中随机抽取200组$\beta_0, \beta_1$值  

```python
# 设置随机种子确保结果可以重复

np.random.seed(84735)

# 根据设定的先验分布，在其中各抽取200个beta_0,200个beta_1, 200个sigma

beta0_200 = np.random.normal(loc = 5, scale = 2, size = 200)
beta1_200 = np.random.normal(loc = 0, scale = 1 , size = 200)
sigma_200 = np.random.exponential(scale=1/0.3, size=200)

# 将结果存在一个数据框内
prior_pred_sample = pd.DataFrame({"beta0":beta0_200,
                                  "beta1":beta1_200,
                                  "sigma":sigma_200})
# 查看抽样结果
prior_pred_sample
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beta0</th>
      <th>beta1</th>
      <th>sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.037555</td>
      <td>-1.138599</td>
      <td>4.569418</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.364096</td>
      <td>-1.004853</td>
      <td>0.771582</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.831791</td>
      <td>-1.189161</td>
      <td>5.563509</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.575321</td>
      <td>0.397816</td>
      <td>5.175152</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.319496</td>
      <td>-1.811424</td>
      <td>4.265667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>2.373954</td>
      <td>0.138251</td>
      <td>7.889629</td>
    </tr>
    <tr>
      <th>196</th>
      <td>4.910970</td>
      <td>0.993390</td>
      <td>11.231550</td>
    </tr>
    <tr>
      <th>197</th>
      <td>5.389999</td>
      <td>1.992606</td>
      <td>4.559478</td>
    </tr>
    <tr>
      <th>198</th>
      <td>5.864917</td>
      <td>-1.374363</td>
      <td>0.147560</td>
    </tr>
    <tr>
      <th>199</th>
      <td>5.951208</td>
      <td>-0.794059</td>
      <td>5.101120</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 3 columns</p>
</div>

在实际操作中，整个取样过程主要依赖于Python中的库和函数。首先，通过`numpy`库中的函数，我们从特定的分布中进行随机抽样。例如，使用`numpy.random.normal`从正态分布中获取样本。然后，抽取到的参数（如$\beta_0$、$\beta_1$）与自变量数据（如$X$）组合成一个数据框（DataFrame）


2. 生成假数据自变量$X$  

* 这里我们根据现实情况来定义X的取值范围  
* 我们使用 `np.arange`设置自变量 Label 的可能取值  
```python
# 通过np.arange设置Label，0 代表 Self， 1 代表other

x_sim = np.array([0, 1])

# 查看自变量值

x_sim
```
array([0, 1])
得到数据后，我们可以通过线性组合的方式计算出μ和σ，从而得到正态分布的两个参数。利用这两个参数，我们就能够从正态分布中进行随机取样，生成符合模型假设的预测数据。在此过程中，我们还可以先观察生成的回归线，检查μ的取值范围，确保生成的数据合理并符合我们的预期。


3. 根据公式 $\mu = \beta_0 + \beta_1 X$ 生成200条回归线, 观察其中的$\mu$是否处在合理的范围内  

* 我们有200次采样，每次采样都有三个参数 beta_0, beta_1, sigma。  

* 结合每次采样的结果，和自变量X，我们可以生成一条直线  

* 重复这个过程200次，我们就能生成200条直线  


在一次随机采样的过程中，我们从β0和β1的分布中随机取值，并生成200次预测数据。通过将每次的β0和β1带入模型，我们可以得到相应的回归线。左图展示了200次采样的结果，显示出预测值在一定范围内变化，并符合我们对β0和β1的设定。右图则展示了特定参数组合下生成的回归线。

> **我们通过一次采样来理解这个过程**  

![Image 6](./img%206.png)  
* **左侧图表**显示了 200 组随机抽取的参数  beta_0, beta_1, sigma 的值。  
  * 这帮助我们直观地看到参数在各自先验分布下的取值范围。  
* **右侧图表**展示了基于抽取的一个特定参数组合绘制的回归线 $Y = \beta_0 + \beta_1 X$  
  * 红色的点表示预测的反应时间在两个条件下的值，蓝色线是连接这两个预测值的回归线。

## 🎯练习1：先验预测  

根据获取的第一条MCMC链的第一组采样参数，结合自变量X的范围，预测 $\mu$ 的值。  

1. 根据回归公式 $\mu = \beta_0 + \beta_1 X$ 预测$\mu$ 的值。  
2. 绘制回归线条。
```python
# 设置随机种子确保结果可以重复

np.random.seed(84735)

# 根据设定的先验分布，在其中各抽取200个beta_0,200个beta_1, 200个sigma

beta0_200 = np.random.normal(loc = 5, scale = 2, size = 200)
beta1_200 = np.random.normal(loc = 0, scale = 1 , size = 200)
sigma_200 = np.random.exponential(scale=1/0.3, size=200)
# 将结果存在一个数据框内
prior_pred_sample = pd.DataFrame({"beta0":beta0_200,
                                  "beta1":beta1_200,
                                  "sigma":sigma_200})
# 查看抽样结果
prior_pred_sample

# 保存为数据框

prior_pred_sample = pd.DataFrame({"beta0": beta0_200, "beta1": beta1_200, "sigma": sigma_200})

# 获取第一组采样参数

beta_0 = prior_pred_sample["beta0"][0]
beta_1 = prior_pred_sample["beta1"][0]

print(f"获取的第一组采样参数值，beta_0:{beta_0:.2f}, beta_1:{beta_1:.2f}")
# ===========================
```
获取的第一组采样参数值，beta_0:6.04, beta_1:-1.14
```python
# 根据回归公式 $\mu = \beta_0 + \beta_1 X$ 预测$\mu$ 的值

# 已知：自变量（标签），self = 1, other = 2

# ===========================
x_sim = np.array([1, 2])
mu = beta_0 + beta_1 * x_sim
print("预测值 μ:", mu)
```
预测值 μ: [4.89895573 3.76035694]
```python
# ===========================

# 绘制回归线，请设置x轴和y轴的变量

# ===========================
x_axis = ...
y_axis = ...

plt.plot(x_axis,y_axis)
plt.xlabel("Label Condition")
plt.ylabel("RT (sec)")  
sns.despine()
```

> 重复上述结果200遍，我们就能得到200次先验预测回归线了  

**可视化先验预测结果**  

* 每一条线代表一次抽样生成的预测，因此我们绘制了200条线。  

* 我们可以观察到 self 和 other 条件下的反应时间如何随着自变量（标签Label）变化。  

* 如果先验设置不合理（如过于宽泛的分布），可能导致预测结果在合理范围之外。  
  * 例如，如果我们将 beta_1 设得过大，可能导致预测的 other 条件下的反应时间显著增加或减少，不符合实验数据的预期。  
  * 因此，通过合理的先验设定，我们可以得到更加符合实验背景的预测结果，这有助于模型对真实数据的拟合。

```python
# 通过 np.array 设置实验条件的取值范围，self=0，other=1

x_sim = np.array([0, 1])

# 设置一个空列表，用来储存每一个的预测结果

mu_outcome = []

# 循环生成 200 次先验预测回归线

for i in range(len(prior_pred_sample)):
    # 根据回归公式计算预测值
    mu = prior_pred_sample["beta0"][i] + prior_pred_sample["beta1"][i] * x_sim
    mu_outcome.append(mu)
```
```python
# 画出每一次的先验预测结果

for i in range(len(mu_outcome)):
    plt.plot(x_sim, mu_outcome[i])

plt.title("prior predictive check")
plt.xlabel("Label Condition")
plt.ylabel("RT (sec)")
sns.despine()
```
![img](./img%207.png)

`课上针对练习的讲解：
绘制200条回归线，我们可以观察到反应时间的预测值分布大致呈现一定的范围。我们特别关注自变量X的取值0和1，因其是我们模型的主要输入。通过汇总这200次预测结果，我们发现反应时间的均值在一个合理范围内，这符合我们的常识判断。首先，反应时间的值始终为正，符合实验条件下的预期；其次，反应时间大致落在0到10之间，未出现不合理的大幅波动，尽管实际实验可能表明反应时间更集中在0到2秒之间，但整体预测结果仍在可以接受的范围内。这样的验证显示，模型的先验分布符合基本常识，不存在明显的偏差。`


### 我们的先验设置的合理吗？  

让我们重新聚焦于我们的研究问题：**在知觉匹配任务中，自我相关信息是否会促进认知加工？** 具体而言，我们探讨自我和他人条件下的认知加工差异，尤其是在反应时间上的表现。  

变量定义：  

* 𝑋：标签（Label）  
  * 在自我匹配范式任务中，标签分为 self 和 other 两种，分别编码为 0 和 1。这些条件用于观察在自我相关和非自我相关条件下的反应时间差异。  

* 𝑌：反应时间（RT）  
  * 表示在 self 或 other 条件下参与者的平均反应时间，通常以秒为单位。我们的目标是通过模型预测 self 和 other 条件下反应时间的差异，并观察其随实验条件的变化。  


在进行先验分布的选择时，虽然每个人的接受度和精细化的需求可能不同，但一般来说，只要先验分布是合理的，并且数据量足够大，贝叶斯更新机制会使得先验逐步接近共同的后验分布。因此，即使每个人选择的先验略有差异，随着数据的积累，这些差异的影响会被逐渐消除，最终后验分布会收敛到一个固定的范围。因此，尽管先验可能不完全一致，但在足够多的数据支持下，不同的先验选择不会对结果产生显著影响。
正如刚才所提到的，通过观察我们得到的结果，可以认为这是合理的：反应时间的变化既有正向也有负向，且反应时间的整体范围符合预期。


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def prior_predictive_plot(beta0_mean=0.5, beta0_sd=0.3, beta1_mean=-0.1, beta1_sd=0.04, sigma_rate=0.2, samples=200, seed=84735):
    """
    生成先验预测图。

    参数：
    - beta0_mean: float，beta0的均值
    - beta0_sd: float，beta0的标准差
    - beta1_mean: float，beta1的均值
    - beta1_sd: float，beta1的标准差
    - sigma_rate: float，控制sigma的指数分布率参数（lambda = 1/scale）
    - samples: int，生成的样本数量
    - seed: int，随机种子，默认为84735，确保结果可重复
    
    输出：
    - 一个先验预测图
    """
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 根据设定的先验分布抽样
    beta0_samples = np.random.normal(loc=beta0_mean, scale=beta0_sd, size=samples)
    beta1_samples = np.random.normal(loc=beta1_mean, scale=beta1_sd, size=samples)
    sigma_samples = np.random.exponential(scale=1/sigma_rate, size=samples)
    
    # 创建数据框存储样本
    prior_pred_sample = pd.DataFrame({
        "beta0": beta0_samples,
        "beta1": beta1_samples,
        "sigma": sigma_samples
    })

    # 定义实验条件（self=0，other=1）
    x_sim = np.array([0, 1])
    
    # 创建一个空列表存储每次模拟的结果
    mu_outcome = []
    
    # 生成先验预测回归线
    for i in range(samples):
        mu = prior_pred_sample["beta0"][i] + prior_pred_sample["beta1"][i] * x_sim
        mu_outcome.append(mu)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    for i in range(len(mu_outcome)):
        plt.plot(x_sim, mu_outcome[i])
    
    plt.title("Prior Predictive Check")
    plt.xlabel("Label Condition")
    plt.ylabel("RT (sec)")
    sns.despine()
    plt.show()

# 使用示例

prior_predictive_plot()
```

## 🎯练习2：先验预测  

🤔请大家判断，下图的先验预测合理吗？  

![Image Name](./img%208.png)
在实际实验中，我们知道被试的反应时间（RT）不会小于0秒。然而，当前模型的先验预测图中可能会包含一些小于0的反应时间，这显然不符合逻辑.  

通过以下练习,你可以尝试对三个参数的先验分布进行设置，观察它们对反应时间预测的影响。
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================

# 请完善代码中...的部分，设置3个参数的值，使先验分布更符合实际情况

# =====================================================

prior_predictive_plot(beta0_mean=...,  # beta 0 的均值
                      beta0_sd=...,    # beta 0 的标准差
                      beta1_mean=...,  # beta 1 的标准差
                      beta1_sd=...,    # beta 1 的标准差
                      sigma_rate=...,  # 控制sigma的指数分布率参数（lambda = 1/scale）
                      samples=200,
                      seed=84735)

```
`课上针对练习的讲解：`
`如果我们遇到另一组数据，是否合理呢？从我的角度来看，这个先验依然是可以接受的。虽然它的分布在0以下出现了一些反应时间值，但这并不意味着先验必须完全符合所有理论预期。实际上，有时只要先验大致合理并且与理论相符，就已经足够。在使用先验预测图时，我们可以直观地看到某些不合理的预测值，意识到它们的存在，并且知道大多数情况下数据是符合预期的。只要有了这种认知，即使存在一些不合理的部分，继续运行模型也是可以接受的。`
## 模型拟合  

### 模型定义  

现在，我们可以结合数据与先验，为参数$(\beta_0, \beta_1, \sigma)$生成后验模型  

* 之后我们可以使用 PyMC 来完成对于模型后验分布的采样过程  

在这之前，我们回顾之前对先验与似然的定义：  

**先验（prior）**  

* > $\beta_{0}   \sim N\left(5, 2^2 \right)$  
  * > 模型的截距项服从均值为 5，标准差为 2 的正态分布。  
* > $\beta_1   \sim N\left(0, 1^2 \right)$  
  * > 模型的斜率项，服从均值为 0，标准差为 1 的正态分布。  
* > $\sigma   \sim \text{Exp}(0.3)$  
  * > 代表误差项的标准差，服从参数为 0.3 的指数分布。  

**似然（likelihood）**  

* > $\mu_i = \beta_0 + \beta_1X_i$  
* > $Y_i {\sim} N\left(\mu_i, \sigma^2\right)$  

在建立模型时，我们使用了刚才讨论的先验分布和似然函数。具体来说，β0采用正态分布，均值为5，标准差为2；β1使用均值为0，标准差为1的正态分布；σ则采用参数为0.3的指数分布。通过这些设置，模型的先验部分就定义完成了。

接下来，我们将这些内容转化为代码。代码的编写其实非常简单，尤其是对于正态分布部分，大家在使用PyMC库时已经相当熟悉了。具体实现时，首先定义先验的分布，并为每个参数命名，比如β0、β1和σ。之后，我们将自变量X定义为数据集中的对应列。

```python
import pymc as pm

with pm.Model() as linear_model:

    # 定义先验分布参数
    beta_0 = pm.Normal("beta_0", mu=5, sigma=2)        
    beta_1 = pm.Normal("beta_1", mu=0, sigma=1)      
    sigma = pm.Exponential("sigma", 3)                    

    # 定义自变量 x
    x = pm.MutableData("x", df['Label'])         

    # 定义 mu，将自变量与先验结合
    mu = beta_0 + beta_1 * x

    # 定义似然：预测值y符合N(mu, sigma)分布
    likelihood = pm.Normal("y_est", mu=mu, sigma=sigma, observed=df['RT_sec'])  # 实际观测数据 y 是 RT
```
在将数据提取出来后，实际上我们只有两个变量：`label` 和 `RT_sec`。接下来，我们根据前面讨论的公式将这两个变量组合起来。
接下来，我们需要将似然函数（likelihood）定义出来。在PyMC中，我们使用 `pm.Normal` 来表示正态分布的似然，其中 `y_est` 代表预测值。对于正态分布而言，参数包括均值 μ 和标准差 σ，因此我们需要将这些参数作为输入。

定义似然函数时，我们的模型会与实际数据进行组合。通过将数据与模型相结合，我们得到了最终的模型形式。在代码中，y_est 表示基于当前参数（即 `μ` 和 `σ`）计算出来的预测值，而我们通过 `observe` 来对比模型预测与实际观察到的数据。


### 后验采样  

1. 接下来我们使用`pm.sample()`进行mcmc采样  

* 我们指定了4条马尔科夫链，保留的采样数为5000，对于每一个参数，在每条链上都有5000个采样结果  

  * $\left\lbrace \beta_0^{(1)}, \beta_0^{(2)}, \ldots, \beta_0^{(5000)} \right\rbrace$  

  * $\left\lbrace \beta_1^{(1)}, \beta_1^{(2)}, \ldots, \beta_1^{(5000)} \right\rbrace$  

  * $\left\lbrace \sigma_1^{(1)}, \sigma_1^{(2)}, \ldots, \sigma_1^{(5000)} \right\rbrace$  

接下来，我们使用 `PyMC` 的 `sample` 方法进行后验采样。通常，设置四条 MCMC 链 `(chains=4) `和 5000 次采样` (draws=5000) `。
```python

# ===========================

# 注意！！！以下代码可能需要运行1-2分钟左右

# ===========================
with linear_model:
    trace = pm.sample(draws=5000,                   # 使用mcmc方法进行采样，draws为采样次数
                      tune=1000,                    # tune为调整采样策略的次数，可以决定这些结果是否要被保留
                      chains=4,                     # 链数
                      discard_tuned_samples=True,  # tune的结果将在采样结束后被丢弃
                      random_seed=84735)

```

```
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [beta_0, beta_1, sigma]
```
```
output()
```
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 5 seconds.
```

在构建模型时，我们通过 `trace` 保存 MCMC 采样的结果。在运行 `PyMC` 时，通常会设置 `5000` 次采样，并指定 `tune` 来进行热身（warm-up），然后设置 `chains=4` 表示使用四条链进行采样。此外，我们还会丢弃热身阶段的样本，并指定 `random_seed` 确保结果的可复现性。完成后，我们将获得一个 `trace`，该 `trace` 包含了所有采样的结果。
在前一节课中，我们讨论了如何从 `trace` 中提取各种信息。这里涉及的关键数据结构是 `inference data`，它使用 `xarray `格式来保存 MCMC 采样的结果。这种数据结构便于存储和操作采样数据，从中提取后验分布、参数估计值及其不确定性等重要信息。



2. 在采样结束之后，我们得到采样样本 trace  

* 后验样本储存在`trace.posterior`中，它的数据类型为 xarray。  

  * 包括了两个维度(索引)，第一个维度为链chain，第二个维度为采样draw  
  
  * 包含了3个变量，即3个参数beta_0, beta_1, sigma  

   	* 我们可以使用 `trace.posterior['beta_0']`提取后验中的$\beta_0$参数  

   	* 我们可以使用 `trace.posterior['beta_0'][0, 10]`提取$\beta_0$第一条链中第10个采样值

**补充 ArviZ inferenceData 介绍**  

`trace` 中的 `inferenceData` 是 PyMC 中用来存储和管理采样结果的格式，专门用于便于分析和可视化。它采用 `ArviZ` 的 `InferenceData` 对象进行封装，包含了多个维度的数据集合，用于存储模型的采样结果、样本统计、后验预测等。以下是 `trace` 中 `inferenceData` 的典型结构和内容：  

1. `posterior`  
`posterior` 是 `InferenceData` 中最核心的一个组，包含了每条链和每次采样的后验分布样本。对于本例子中的模型，`posterior` 包括以下参数：  

* `beta_0`: 表示截距参数的后验分布采样值。  
* `beta_1`: 表示斜率参数的后验分布采样值。  
* `sigma`: 表示残差标准差的后验分布采样值。  

数据结构通常为 `(chain, draw, parameter)`，其中 `chain` 表示不同的链，`draw` 是每条链的样本数量。我们可以用 `posterior` 中的数据来绘制各个参数的后验分布图，分析它们的均值、中位数及不确定性。  

2. `sample_stats`  
`sample_stats` 存储每次采样的统计信息，帮助评估采样质量。这些数据有助于诊断 MCMC 链的收敛性和性能。常见的统计信息包括：  

* `lp`: 对数后验密度 (log-posterior)，用于评估样本的相对“好坏”。  
* `r_hat`: 用于评估链的收敛情况，理想值接近1，若远离1可能表明链未收敛。  
* `ess_bulk` 和 `ess_tail`: 分别为整体和尾部的有效样本数，用于衡量采样的独立性。  
* `diverging`: 若存在divergence，通常表明模型的特定区域无法很好地被估计。  

这些统计信息可以用于诊断和调整模型的采样参数，比如增大 `tune` 或 `draws` 的数量。  

3. `observed_data`  
`observed_data` 包含实际的观测数据，这在本例中为 `df['RT_sec']`。在模型的 `likelihood` 部分指定了 `observed=df['RT_sec']`，因此它也会被存储到 `inferenceData` 中，便于对比观测值和预测值。  

1. `posterior_predictive`  
如果在模型中使用 `pm.sample_posterior_predictive` 来生成后验预测分布，这些预测值会存储在 `posterior_predictive` 中。`posterior_predictive` 包含每条链和每次采样生成的预测值（`y_est`），方便与观测数据（`observed_data`）进行对比，评估模型的预测效果和拟合度。  

1. `prior`  
若通过 `pm.sample_prior_predictive` 生成了先验预测分布，`prior` 会保存这些值。它能帮助我们了解模型在没有观测数据时的先验分布，为参数的选择提供参考。  

<div style="padding-bottom: 30px;"></div>
在运行完采样（sampling）之后，得到的结果是一个trace数据，这个数据通常是交互式的，尤其是在Jupyter Notebook环境中。Jupyter提供了可视化功能，帮助用户更好地理解数据。但在纯代码环境中，这种交互式功能就无法实现，因此需要注意保存和读取操作。

Trace数据通常包含模型的后验分布信息以及观测数据。没有进行采样时，模型并没有后验数据。定义了模型并加入数据后，才会生成相应的观测数据。在这种情况下，trace数据被存储为inference data格式，这通常是一个xarray格式的数据集（dataset）
>💡数据演示见课件

**trace***
在PyMC中，trace数据的格式称为inference data，这种格式是多维的，能够有效存储模型的各类信息，尤其是采样后的后验分布结果。inference data包含了两个重要概念：维度和坐标。

维度定义了模型中变量的特征，例如反应时（RT）等。如果模型包含多个变量（如反应时、决策时间或EEG数据），每个变量都可以有不同的维度。例如，可能有105个维度对应105个数据点。
坐标用于指定每个维度的索引。坐标值可以是数字（如0, 1, 2...104），也可以是字符串（如学校名称），以便于更清晰地标识和索引数据。
通过维度和坐标的结合，inference data允许我们处理多维数据，甚至在变量之间进行索引和提取。每个索引对应一个数据值。例如，在模型中，反应时的数据会根据相应的维度坐标来展示。

在Jupyter Notebook中，inference data的显示是交互式的，可以方便地展开查看各个数据变量。当没有采样结果时，数据结构只包含观测变量。当进行后验采样后，inference data会新增一个“后验”组，包含后验分布的维度和坐标，以及与之对应的采样值。


**trace.posterior**
trace.posterior['beta_0']
trace.posterior['beta_0'][0, 10]

inference data 中的 group 名称是**固定**的，这样的设计有其优势和局限性。其主要优点在于统一性和简化操作。当进行贝叶斯分析时，常见的几个 group 名称通常会被使用，最常用的就是 posterior 和 posterior predictive，它们在不同的贝叶斯模型中会始终保持一致。例如：

- **Posterior：**这是贝叶斯分析中最常用的数据，通常是我们关注的重点。我们会基于后验分布对模型参数进行分析，包括提取数据、绘图等操作。
- **Posterior Predictive：**这是基于后验分布进行预测的结果，通常用于模型的评估和检验，或者生成对比不同模型的预测。

固定的 group 名称帮助用户熟悉和简化操作流程，因为不论使用何种贝叶斯模型，这些 group 名称都是一致的，避免了每次使用时重新定义。这种命名方式的好处是它使得贝叶斯分析中的数据处理更加一致和方便，尤其是在进行模型比较时。

不过，这种固定命名方式也有局限性，因为我们无法随意更改这些 group 的名称。如果有特定需求需要重新定义或修改某些数据结构的名称，可能会受到限制。尽管如此，一旦理解了这些常用 group 的含义，使用起来就会变得更加顺手和高效。

### MCMC 诊断  

使用`az.plot_trace()`可视化参数的后验分布
```python
import arviz as az

ax = az.plot_trace(trace, figsize=(7, 7),compact=False,legend=True)
plt.tight_layout()
plt.show()
```
![img 9](./img%209.png)

从图中可以清晰地观察到，每条链使用不同的颜色区分。对于诊断，我们重点关注链条是否充分混合。若如图所示，不同链条的轨迹完全混合在一起，且从不同起点出发后最终表现出类似的分布，这说明采样质量较高，链条运行良好。

对于模型的诊断信息 ess_bulk 和 r_hat (当然你可以结合可视化进行诊断)。  
* 其中，各参数的 r_hat 均接近于 1；  
* 各参数的ess_bulk 均大于 400，并且有效样本量占比 (6000/20000)=0.3，大于0.1(即10%)。 

```python 
az.summary(trace, kind="diagnostics")
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>beta_0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>10224.0</td>
      <td>11211.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>beta_1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>10341.0</td>
      <td>11605.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>11501.0</td>
      <td>11301.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



除了可视化诊断外，我们还可以通过量化指标对 MCMC 的结果进行评估，例如有效样本量（ESS）和 R-hat 指标。这同样可以通过一行代码完成，即使用 `az.summary(trace)` 函数。若仅关注诊断信息，可以通过设置 `kind="diagnostics"` 参数，使输出更简洁，专注于相关指标。

从结果中可以看到，R-hat 指标接近 1，这是链条充分混合、收敛性良好的标志。同时，ESS 指标也很高（如 10000+），说明有效样本量充足，采样结果具有较高的信赖度。这表明我们的后验分布和 MCMC 链的质量较高。

需要注意的是，由于模型较为简单，即使将采样次数减少到 2000 次，可能仍然能够得到类似的高质量结果。简单模型通常更容易快速达到收敛。




## 后验预测  


在确认 MCMC 采样结果可靠后，下一步需要检验模型与数据的匹配程度。之前我们评估的是参数估计的稳定性，即 MCMC 链是否收敛；但模型是否适合数据尚未验证。这一步的目标是判断模型对数据的拟合效果是否良好。

MCMC 诊断仅能显示采样过程的收敛性和采样质量，而不能直接说明模型在观测数据上的拟合效果或预测能力。为评估模型的预测能力，我们可以进行**后验预测检查**，通过模拟新的数据样本来比较模型生成的数据与实际观测数据的分布是否一致，从而验证模型的合理性和适用性。  

* 为了更全面地反映模型的不确定性，我们可以基于 20000 对参数值生成 20000 条回归线。  

* 这些回归线将展示在 "self" 和 "other" 条件下的预测差异，以及模型在不同参数样本下的预测范围。这种基于后验参数的预测被称为后验预测 (Posterior Prediction)。  

* 在进行后验预测时，我们利用模型后验采样得到的参数进行预测，并结合真实数据生成预测值的分布。  
* 这个过程不仅可以帮助我们检查模型对数据的适配度，还能通过可视化展现预测的不确定性。

一个好的模型，其核心特征在于与数据的拟合程度良好。模型根据观测数据估计出的参数，应该能够生成与实际观测数据高度相似的新数据。如果新生成的数据在分布、趋势等方面与原始观测数据非常一致，或者误差极小，就可以认为模型与数据之间的匹配是令人满意的。
在模型匹配评估中，有一种非常直观且符合逻辑的方式：首先，根据观测数据拟合出模型并得到参数的后验分布；然后，利用这些后验分布的参数生成模拟数据；最后，将生成的数据与原始观测数据进行对比，观察二者在特性上的相似性。如果生成的数据能够很好地再现原始数据的特性，就说明模型与数据的拟合效果较好。
后验预测(Posterior Predictive Check)正是完成这一任务的重要步骤。在完成后验采样并确认 MCMC 链的收敛性和采样质量之后，仍需通过后验预测进一步验证模型的合理性。具体而言，通过 MCMC 采样获得的后验分布，我们可以提取大量的参数组合，用以生成模拟数据。例如，假设有 2 万组后验参数组合，这相当于生成了 2 万条回归线。这些回归线是否能够捕捉到观测数据中的特性，例如变量间的差异，是模型是否合理的重要评估标准。

### 🎯练习  

根据 **先验预测检验可视化预测结果**的思路，对于后验预测结果进行可视化。  

1. 使用真实数据中的自变量Label  
2. 根据 20000对参数（beta_0, beta_1），与自变量(Label)进行组合，生成了20000条回归线  
3. 绘制后验预测结果  

```python
# 通过np.arange设置 x 轴代表 Label，其中 0 代表 Self， 1代表other

x_sim = np.array([0, 1])

# 选取一组参数来作为预测

beta_0 = trace.posterior["beta_0"].stack(sample=("chain", "draw"))[:2]
beta_1 = trace.posterior["beta_1"].stack(sample=("chain", "draw"))[:2]

# 生成20000条回归线

y_sim_re = beta_0 + beta_1 * x_sim

# 绘制真实数据的散点图

plt.scatter(trace.constant_data.x, trace.observed_data.y_est,c="r", label="observed data")

# 绘制回归线条

plt.plot(x_sim, y_sim_re, c="grey", label = "Predicted Mean")
plt.scatter(x_sim, y_sim_re, color="black", s=50)

# 设置标题等

plt.xlim(-0.5, 1.5)
plt.xticks([0, 1])
plt.title("posterior predictive check")
plt.xlabel("Label")
plt.ylabel("RT (sec)")  
plt.legend()
sns.despine()

```

![img 10](./img%2010.png)

在后验预测中，我们绘制的数据实际上是预测值的均值。这与线性回归中的概念类似，即模型预测的是在某个特定自变量$X$取值时，因变量$Y$的期望均值。实际观测的数据通常会围绕这个均值上下波动。绘图中，黑色点显示了预测的均值，实际数据的分布则展现了围绕均值的波动特征。

**使用`plot_lm` 绘制后验预测的线性模型**  

**代码详解**  

* 与上一段代码最大的不同之处在于，此时需要将`y_model` 存入`trace`中  

* 在`az.plot_lm`中:  
  * `y` 为真实数据中的因变量`df.RT_sec`  
  * `x` 为真实数据中的自变量`df.Label`  
  * `y_model` 为结合后验采样生成的预测值  
    （在图中表示为黄色和灰色的回归线）  

> 😎*跑起来快很多*


```python
import xarray as xr

# 导入真实的自变量

x_value = xr.DataArray(df.Label)

# 基于后验参数生成y_model

trace.posterior["y_model"] = trace.posterior["beta_0"] + trace.posterior["beta_1"] * x_value
df['Mean RT'] = df.groupby['Label']('RT_sec').transform('mean')

# 绘制后验预测线性模型

az.plot_lm(
           y= df['Mean RT'],
           x= df.Label,
           y_model = trace.posterior["y_model"],
           y_model_mean_kwargs={"color":"black", "linewidth":2},
           figsize=(6,4),
           textsize=16,
           grid=False)

# 设置坐标轴标题、字体大小

plt.xlim(-0.5, 1.5)
plt.xticks([0, 1])
plt.xlabel('Label')  
plt.ylabel('RT (sec)')  
plt.legend(['observed mean', 'Uncertainty in mean', 'Mean'])

sns.despine()
```

![img 11](./img%2011.png)

在后验预测中，如果我们绘制真实数据的均值，就能更直观地观察预测的合理性。通过对比实际观测均值和预测均值，可以发现预测结果通常较为贴合实际数据分布。同时，每一组MCMC采样参数都可以生成一组预测，因此理论上可以产生大量预测进行比较。
绘图中的预测均值相当于正态分布中的$μ$，每条线代表不同自变量$X$取值（如$X=0$ 和 $X=1$）时的预测均值。此外，基于这些预测均值$μ$和观测数据中相应的标准差$σ$，我们还可以生成模拟的随机观测数据（如反应时）。

### 通过MCMC采样值理解后验预测分布  

* 通过MCMC采样，三个参数各获得了20000个采样值$\left(\beta_0^{(i)},\beta_1^{(i)},\sigma^{(i)}\right)$  

* 根据 20000 组参数值 $\beta_0$ 和 $\beta_1$，可以得到 20000 个均值 $\mu$ 的可能值。然后再根据 $\mu$ 生成预测值 $Y_{\text{new}}$。  
* 20000 个均值 $\mu$ 构成了预测的均值分布：  

$$  
\left[  
\begin{array}{ll}  
\beta_0^{(1)} & \beta_1^{(1)} \\  
\beta_0^{(2)} & \beta_1^{(2)} \\  
\vdots & \vdots \\  
\beta_0^{(20000)} & \beta_1^{(20000)} \\  
\end{array}  
\right]  
\;\; \longrightarrow \;\;  
\left[  
\begin{array}{l}  
\mu^{(1)} \\  
\mu^{(2)} \\  
\vdots \\  
\mu^{(20000)} \\  
\end{array}  
\right]  
$$  

* 为了模拟这个过程，我们首先从后验分布中提取采样结果，并生成每个采样值对应的预测均值 $\mu$。每个均值 $\mu^{(i)}$ 可以通过以下公式计算：  

$$  
\mu^{(i)} = \beta_0^{(i)} + \beta_1^{(i)} X  
$$  

* 然后，在每个均值 $\mu^{(i)}$ 的基础上，加入噪声项 $\epsilon$ 来生成 $Y_{\text{new}}^{(i)}$：  

$$  
Y_{\text{new}}^{(i)} = \mu^{(i)} + \epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^{(i)})  
$$  

* 这里，$\epsilon^{(i)}$ 是服从均值为 0，方差为 $\sigma^{(i)}$ 的正态分布。  

> 可以注意到，生成的预测值受到两种变异的影响：  
>
> * 一是参数估计的不确定性（即 $\beta_0$ 和 $\beta_1$ 的后验分布带来的变异），导致不同样本的均值 $\mu$ 具有差异；  
> * 二是随机误差项 $\epsilon$ 的影响，使得在相同均值 $\mu$ 下生成的预测值 $Y_{\text{new}}$ 仍然存在随机波动。这两种变异共同决定了最终预测值的后验预测分布。  

<div style="padding-bottom: 30px;"></div>

在MCMC采样或贝叶斯统计中，我们可以通过采样得到的2万组参数来生成预测值。例如，对于每组参数$\beta_0$、$\beta_1$，结合实际数据中的$X_i$值，通过公式$\beta_0 + \beta_1 · X$，就可以计算出一系列预测均值。这些预测均值实际上构成了一个完整的回归模型，是参数和数据的线性组合。
同时，从采样结果中也可以提取对应的$σ$值，与这些均值配合，用来表征预测分布的波动范围。这种方式让我们能够更精确地构建每组采样参数下的预测分布，进一步检验模型对数据的拟合程度。

### 提取后验样本并生成预测  

**我们也可以用代码来进行模拟，首先我们先进行单次完整的抽取过程**

```python

# 采样得到的参数后验分布都储存在 trace.posterior中，我们进行一些提取操作

pos_sample = trace.posterior.stack(sample=("chain", "draw"))

# 将每个参数的20000次采样结果存储在数据框中

df_pos_sample = pd.DataFrame({"beta_0": pos_sample["beta_0"].values,
                              "beta_1": pos_sample["beta_1"].values,
                              "sigma": pos_sample["sigma"].values})

# 查看参数

df_pos_sample
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beta_0</th>
      <th>beta_1</th>
      <th>sigma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.711816</td>
      <td>0.197405</td>
      <td>0.227486</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.688240</td>
      <td>0.157603</td>
      <td>0.235002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.675064</td>
      <td>0.186001</td>
      <td>0.237489</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.683199</td>
      <td>0.189842</td>
      <td>0.212226</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.712327</td>
      <td>0.133783</td>
      <td>0.216075</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19995</th>
      <td>0.698971</td>
      <td>0.127975</td>
      <td>0.213383</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>0.679192</td>
      <td>0.165039</td>
      <td>0.206986</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>0.660000</td>
      <td>0.195370</td>
      <td>0.260113</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>0.687715</td>
      <td>0.117434</td>
      <td>0.210055</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>0.732075</td>
      <td>0.184231</td>
      <td>0.249545</td>
    </tr>
  </tbody>
</table>
<p>20000 rows × 3 columns</p>
</div>

```python
# 抽取第一组参数组合，生成正态分布的均值

row_i = 0  
X_i = 1
mu_i = df_pos_sample.beta_0[row_i] + df_pos_sample.beta_1[row_i] * X_i
sigma_i = df_pos_sample.sigma[row_i]

# 从正态分布中随机抽取一个值，作为预测值

prediction_i = np.random.normal(
                                loc = mu_i,
                                scale= sigma_i,
                                size=1)

# 你可以运行该代码块多次，比较在相同参数下，预测值的变化(感受采样变异)

print(f"mu_i: {mu_i:.2f}, 预测值：{prediction_i[0]:.2f}")
```
`
mu_i: 0.91, 预测值：1.12
`

在贝叶斯统计或MCMC中，通过采样得到的每组参数（如$\mu$和$\sigma$）用于生成预测数据。对于每一个给定的$X$值，我们可以计算出对应的$\mu$和$\sigma$，然后从正态分布中随机采样，得到一个预测的反应时间或观测数据。这个预测值是基于当前参数（$\mu$ 和 $\sigma$）从分布中抽取的随机样本。
通过这种方式，我们可以进行多次采样。例如，对于每一组$\mu$和$\sigma$，我们可以进行多个观测值的生成。假如每组$\mu$和$\sigma$生成5个观测值，经过2万个参数组合的迭代，我们将生成10万个模拟数据。这些模拟数据可以与实际的观测数据进行比较，从而评估模型预测的准确性和数据拟合的程度。


**使用代码模拟多次后验预测**  

* 通过上述四行代码，我们已经进行了一次完整的后验预测  

* 我们可以写一个循环，重复这个过程20000次  

* 最后的结果中，每一行代表一个参数对；mu 为预测的均值，y_new 为实际生成的预测值。
```python
# 生成两个空列，用来储存每一次生成的均值mu，和每一次抽取的预测值y_new

df_pos_sample['mu'] = np.nan
df_pos_sample['y_new'] = np.nan
X_i = 1
np.random.seed(84735)

# 将之前的操作重复20000次

for row_i in range(len(df_pos_sample)):
    mu_i = df_pos_sample.beta_0[row_i] + df_pos_sample.beta_1[row_i] * X_i
    df_pos_sample["mu"][row_i] = mu_i
    df_pos_sample["y_new"][row_i] = np.random.normal(loc = mu_i,
                                            scale= df_pos_sample.sigma[row_i],
                                            size=1)
df_pos_sample
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>beta_0</th>
      <th>beta_1</th>
      <th>sigma</th>
      <th>mu</th>
      <th>y_new</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.711816</td>
      <td>0.197405</td>
      <td>0.227486</td>
      <td>0.909221</td>
      <td>1.027236</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.688240</td>
      <td>0.157603</td>
      <td>0.235002</td>
      <td>0.845844</td>
      <td>0.771125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.675064</td>
      <td>0.186001</td>
      <td>0.237489</td>
      <td>0.861065</td>
      <td>1.078580</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.683199</td>
      <td>0.189842</td>
      <td>0.212226</td>
      <td>0.873041</td>
      <td>1.040203</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.712327</td>
      <td>0.133783</td>
      <td>0.216075</td>
      <td>0.846110</td>
      <td>0.772590</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19995</th>
      <td>0.698971</td>
      <td>0.127975</td>
      <td>0.213383</td>
      <td>0.826947</td>
      <td>1.027230</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>0.679192</td>
      <td>0.165039</td>
      <td>0.206986</td>
      <td>0.844231</td>
      <td>1.072505</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>0.660000</td>
      <td>0.195370</td>
      <td>0.260113</td>
      <td>0.855370</td>
      <td>0.783391</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>0.687715</td>
      <td>0.117434</td>
      <td>0.210055</td>
      <td>0.805150</td>
      <td>0.811509</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>0.732075</td>
      <td>0.184231</td>
      <td>0.249545</td>
      <td>0.916306</td>
      <td>0.986667</td>
    </tr>
  </tbody>
</table>
<p>20000 rows × 5 columns</p>
</div>



**绘制后验预测分布**  

根据刚刚生成的数据，我们可以分别绘制出 $\mu$ 与 $Y_{new}$ 的后验预测分布图
```python
# 查看真实数据中的取值，与后验预测分布作对比
df2 = df.drop(["Mean RT"],axis=1).copy()
print("x=1时y的取值有:", np.array(df2[df2["Label"]==1]))
```

x=1时y的取值有: [[1.    0.753]
 [1.    0.818]
 [1.    0.917]
 [1.    0.717]
 [1.    0.988]
 [1.    0.95 ]
 [1.    0.657]
 [1.    0.829]
 [1.    1.143]
 [1.    0.756]
 [1.    0.665]
 [1.    0.846]
 [1.    0.839]
 [1.    0.914]
 [1.    0.712]
 [1.    1.33 ]
 [1.    0.786]
 [1.    0.626]
 [1.    0.912]
 [1.    0.725]
 [1.    0.956]
 [1.    0.485]
 [1.    1.417]
 [1.    0.604]
 [1.    0.789]
 [1.    1.327]
 [1.    1.357]
 [1.    0.635]
 [1.    0.871]
 [1.    1.287]
 [1.    0.739]
 [1.    1.331]
 [1.    0.907]
 [1.    1.015]
 [1.    1.125]
 [1.    0.868]
 [1.    0.582]
 [1.    1.233]
 [1.    1.03 ]
 [1.    0.791]
 [1.    1.028]
 [1.    0.918]
 [1.    0.793]
 [1.    0.909]
 [1.    0.646]
 [1.    0.467]
 [1.    0.843]
 [1.    0.61 ]
 [1.    0.972]
 [1.    0.851]
 [1.    1.208]
 [1.    0.473]
 [1.    0.407]
 [1.    1.416]
 [1.    1.164]
 [1.    0.605]
 [1.    1.071]
 [1.    0.425]
 [1.    0.634]
 [1.    0.393]
 [1.    1.02 ]
 [1.    0.414]
 [1.    0.698]]

这里展示了通过手动编写代码来生成预测数据的过程，最终得到了$\mu$和$\sigma$以及对应的预测数据$Y_i$。虽然这种方法能够完成任务，但代码较长且可能会存在一些瓶颈。值得注意的是，现代的工具如ArviZ已经为我们提供了简便的一行代码，能够自动化地处理这一过程。


```python

# 新建画布
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, sharex=True)

# 在第一个画布中绘制出生成的mu的分布
sns.kdeplot(data=df_pos_sample,
            x="mu",
            color="black",
            ax=axs[0])

# 在第二个画布中绘制出生成的y_new的分布
sns.kdeplot(data=df_pos_sample,
            x="y_new",
            color="black",
            ax=axs[1])

fig.suptitle('Posterior predictive distribution(x=1)', fontsize=15)
sns.despine()
```
![img 12](./img%2012.png)

从上图可以看到， $Y_{new}$ 分布的不确定性远大于 $\mu$ 分布的不确定性：  

* $\mu$ 分布窄且集中，反映了模型的稳定预测中心；  
* 而 $Y_{new}$ 分布较宽，反映了模型的不确定性。  

> 正如之前提到那样，生成的预测值受到两种变异的影响：  
>
> * 一是参数估计的不确定性（即 $\beta_0$ 和 $\beta_1$ 的后验分布带来的变异），导致不同样本的均值 $\mu$ 具有差异；  
> * 二是从分布到数据中，另一个参数 $\sigma$ 的影响，进一步放大了预测值 $Y_{\text{new}}$ 的不确定性。这两种变异共同决定了最终预测值的后验预测分布。
>
### 总体后验预测分布  

* 除了生成特定自变量下，因变量的分布，也可以生成总体因变量的后验预测分布  

* 通过 `pymc.sample_posterior_predictive`方法可以快速从模型生成后验预测数据。  
* 可以看到 ppc_data 中多了一项 posterior_predictive

```python
with linear_model:
    ppc_data = pm.sample_posterior_predictive(trace)

ppc_data
```
```
Sampling: [y_est]
```

在前面，我们手动编写了后验预测的过程，但实际上，在PyMC中已经提供了一个非常方便的算法，即 `pymc.sample_posterior_predictive`。只需要传入 `trace` 参数，运行这行代码后，它会自动生成一个新的组，名为 `sample posterior predictive`。


接着，我们可以使用 arviz 提供的后验预测检查函数 `plot_ppc`来绘制结果。  

* 黑色线条代表观测值总体的分布情况。  

* 蓝色线代表每一对采样参数对应的后验预测的分布情况。  

* 橙色为后验预测的均值的分布情况
```python
# num_pp_samples 参数代表从总的采样(20000)选取多少采样(这里是1000)进行后验预测计算

az.plot_ppc(ppc_data, num_pp_samples=1000)
```
```
<Axes: xlabel='y_est'>
```
![img 13](./img%2013.png)

运行后，可以通过ArviZ的`plot PPC`绘制预测观测值与实际观测值的关系。由于有2万个预测值，直接绘制会显得非常混乱，因此通常选择只画其中一部分，通常为500条或1000条。蓝色线条表示每组MCMC参数生成的预测线，每组参数会生成与观测值数量相同的数据点，绘制出对应的密度曲线。通过这些曲线，能够看出模型的整体分布情况，黄色虚线代表这些曲线的均值，反映了一个近似正态分布。然而，实际观测值（黑色曲线）与预测结果之间存在偏差，表明数据并不完全符合正态分布。尽管如此，这个模型在大致趋势上与观测数据相符，但进一步的改进可能涉及更适合反应时间（RT）分布的模型，如广义线性模型。

### 对新数据的预测  

* 采样得到的后验参数基于编号为"201"的被试数据，到目前为止，我们都在使用后验参数对这一批数据做出后验预测  

* 那么基于编号为"201"的被试数据得出的后验参数估计对其他数据的预测效果如何？  

* 我们可以选用一批新的数据，查看当前参数是否能预测新数据(例如 "205")中的变量关系

```python
# 筛选编号为“205”的被试的数据

df_new = df_raw[(df_raw["Subject"] == "205") & (df_raw["Matching"] == "Matching")]

# 选择需要的两列

df_new  = df[["Label", "RT_sec"]]

# 设置索引
df_new["index"] = range(len(df_new))
df_new = df_new.set_index("index")

# 显示部分数据

df_new.head()

```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>RT_sec</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.753</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.818</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.717</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.988</td>
    </tr>
  </tbody>
</table>
</div>

```python
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def plot_posterior_predictive(df, trace, ax=None, title = "Posterior Predictive"):
    """
    绘制后验预测线性模型，展示不同 Label 条件下的平均反应时间 (RT) 及其不确定性。

    参数:
    - df : pandas.DataFrame
        包含实验数据的 DataFrame，其中需要包括 'Label' 和 'RT_sec' 两列。
    - trace : arviz.InferenceData
        包含后验参数的 ArviZ InferenceData 对象，需要包括 `beta_0` 和 `beta_1`。
    - ax : matplotlib.axes.Axes, optional
        用于绘制图像的 matplotlib 轴对象。如果未提供，将自动创建一个新的轴对象。
        
    Returns:
    - ax : matplotlib.axes.Axes
        返回绘制了图形的 matplotlib 轴对象。
    
    说明:
    该函数首先将 `Label` 列转换为 xarray 数据格式，以用于生成后验预测模型。接着，
    基于后验参数 `beta_0` 和 `beta_1` 计算模型预测的 `y_model`，并对每个 `Label`
    组内的反应时间 (`RT_sec`) 计算均值。在此基础上，使用 ArviZ 的 `plot_lm` 绘制
    后验预测线性模型，并设置图例、坐标轴范围、标签和其他样式。
    """
    # 如果没有提供 ax，则创建新的图形和轴对象
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # 导入真实的自变量
    x_value = xr.DataArray(df.Label)

    # 基于后验参数生成 y_model
    trace.posterior["y_model"] = trace.posterior["beta_0"] + trace.posterior["beta_1"] * x_value
    df['Mean RT'] = df.groupby('Label')['RT_sec'].transform('mean')

    # 绘制后验预测线性模型
    az.plot_lm(
        y=df['Mean RT'],
        x=df.Label,
        y_model=trace.posterior["y_model"],
        y_model_mean_kwargs={"color":"black", "linewidth":2},
        textsize=16,
        grid=False,
        axes=ax  # 使用传入的轴对象
    )

    # 设置坐标轴标题、范围和字体大小
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_ylim(0.65, 0.95)
    ax.set_xlabel('Label')
    ax.set_ylabel('RT (sec)')
    ax.legend(['observed mean', 'Uncertainty in mean', 'Mean'])
    ax.set_title(title)

    # 去除顶部和右侧边框
    sns.despine(ax=ax)

    # 返回轴对象
    return ax
```


除了对旧数据进行预测，我们还可以对新的数据进行预测。例如，当我们有一组新的被试数据时，包含一系列的试次（trials），我们可以利用已建立的模型来预测该被试的反应效果。

```python
fig, axes = plt.subplots(1,2, figsize=(9,4))

plot_posterior_predictive(df, trace, axes[0], "Subject 201")
plot_posterior_predictive(df_new, trace, axes[1], "Subject 205")

plt.tight_layout()
plt.show()
```
![img 14](./img%2014.png)
对于某个被试（如被试201），我们可以看到他的训练数据的预测效果很好，黑线和观测均值几乎完全重合，说明预测非常准确。然而，当我们尝试预测另一个新的被试的数据时，效果可能不如前者。具体来说，新的观测数据并不完全位于预测曲线的中央，表明预测与实际观测之间存在一定的偏差。尽管如此，预测仍然能够在一定程度上捕捉到数据的趋势，且大部分情况下，新的被试的均值仍然落在预测的范围内。后续在讲解层级模型时，我们将讨论如何将多个被试的数据联合起来，进行更精确的预测。


## 后验推断  

我们共得到20000对$\beta_0$和$\beta_1$值，可以通过`az.summary()`总结参数的基本信息  

* 此表包含了模型的诊断信息，例如参数的均值、标准差和有效样本大小（ess_bulk 和 ess_tail）。  
* 还提供了每个参数的 94% 最高密度区间（HDI），用于展示参数的不确定性范围。

我们刚才主要通过后验预测检验评估了模型，结果显示模型勉强可以接受。接下来，我们可以继续使用该模型和它的参数进行推断，以解决最初提出的问题：自我条件与他人条件是否存在差异。在进行推断时，我们关注的是参数的不确定性，即这些参数值是否与0有显著的差异。

```python
az.summary(trace)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>beta_0</th>
      <td>0.712</td>
      <td>0.036</td>
      <td>0.645</td>
      <td>0.780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10456.0</td>
      <td>11282.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>beta_1</th>
      <td>0.148</td>
      <td>0.046</td>
      <td>0.061</td>
      <td>0.235</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10272.0</td>
      <td>11248.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.231</td>
      <td>0.016</td>
      <td>0.202</td>
      <td>0.262</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12876.0</td>
      <td>12118.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_model[0]</th>
      <td>0.860</td>
      <td>0.029</td>
      <td>0.806</td>
      <td>0.916</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21700.0</td>
      <td>15504.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_model[1]</th>
      <td>0.860</td>
      <td>0.029</td>
      <td>0.806</td>
      <td>0.916</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21700.0</td>
      <td>15504.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>y_model[100]</th>
      <td>0.712</td>
      <td>0.036</td>
      <td>0.645</td>
      <td>0.780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10456.0</td>
      <td>11282.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_model[101]</th>
      <td>0.712</td>
      <td>0.036</td>
      <td>0.645</td>
      <td>0.780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10456.0</td>
      <td>11282.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_model[102]</th>
      <td>0.712</td>
      <td>0.036</td>
      <td>0.645</td>
      <td>0.780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10456.0</td>
      <td>11282.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_model[103]</th>
      <td>0.712</td>
      <td>0.036</td>
      <td>0.645</td>
      <td>0.780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10456.0</td>
      <td>11282.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>y_model[104]</th>
      <td>0.712</td>
      <td>0.036</td>
      <td>0.645</td>
      <td>0.780</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10456.0</td>
      <td>11282.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 9 columns</p>
</div>

我们关注的参数实际上是β1，因此可以通过使用`summary`函数来查看其详细信息。前面提到过，`summary`函数如果不加上`kind="diagnostics"`，会提供更丰富的内容，比如β0、β1和σ的均值、标准差（SD）以及它们的94%高密度区间（HDI），前面也有3%到97%之间，而不是严格的95%。


* 我们可以使用均值来理解生成的后验分布，通过上表我们知道  

 	* $\beta_0$ 表示 self 条件下的基准反应时间约为 0.796 秒。  

 	* $\beta_1$ 表示 self 和 other 条件下的反应时间差异非常小，几乎可以忽略不计。  

* 注意：尽管表中显示了参数的均值，但这些均值只是后验分布的一个概要信息。  
 	* 我们还可以从 HDI 和标准差中观察到后验分布的广泛性，反映了模型的内在不确定性。  
 	* 因此，仅使用均值生成的回归线并不足以充分展示后验分布的复杂性和不确定性。


我们更关注的是$β_1$，因为它表示自我和他人之间的差异。根据结果，β1的区间从0.06到0.235，意味着自我和他人之间的反应时间差异在60毫秒到200毫秒之间。那么，这个差异算大还是小呢？通常，在认知实验中，20毫秒甚至10毫秒的差异就被认为是有意义的。因为反应本身非常快速，所以即使是肉眼看起来不明显的差异，也可能具有统计上的意义，尽管没有办法凭直观做出规范的推断。


上节课我们学习了使用 HDI + ROPE 进行检验。在这里我们假设 ($\beta_1$) 的值在 $[-0.05, 0.05]$ 范围内可以视为实用等效，  

即如果$\beta_1$落在这个范围内，说明 self 和 other 条件之间的反应时间差异可以忽略不计，从而在实践上认为两者无显著差异。  

1. **ROPE 区间**：我们设定 $[-0.05, 0.05]$ 为 ROPE 区间，表示 self 和 other 条件下的反应时间差异在此范围内被视为无显著差异。该范围表示了对“等效零效应”的假设，即认为微小的差异在实践中可以忽略。  

2. **HDI (Highest Density Interval)**：后验分布的 95% 最高密度区间（HDI）显示了 $\beta_1$ 的不确定性范围，帮助我们了解后验分布中最可信的值区域。  

3. **结果解读**：  
   * 如果 $\beta_1$ 的后验分布大部分位于 ROPE 区间内，我们可以认为 self 和 other 条件下的反应时间差异在实用上无显著意义，即这两种条件在反应时间上几乎等同。  
   * 如果后验分布的很大一部分超出了 ROPE 区间，则表明 self 和 other 条件之间的差异在实用上具有显著性，值得进一步关注。  




规范推断通常是使用大家认可的方法，比如HDI（Highest Density Interval）加上ROPE（Region of Practical Equivalence）。ROPE的定义是，当两个条件之间的差距在±50毫秒以内时，我们认为它们没有差异。ROPE是一个比较保守的标准，意味着只有当差距超出这个区间，才认为两个条件之间有实际差异。在这个例子中，如果我们设定ROPE为±50毫秒，那么它对应的区间就是-0.05到+0.05。

一旦设定了ROPE，我们可以通过ArviZ中的一行代码，使用plot HDI和ROPE功能，来查看HDI与ROPE之间的重叠情况，从而评估模型的差异是否显著。

```python
import arviz as az

# 定义 ROPE 区间，根据研究的需要指定实际等效范围

rope_interval = [-0.05, 0.05]

# 绘制后验分布，显示 HDI 和 ROPE

az.plot_posterior(
    trace,
    var_names="beta_1",
    hdi_prob=0.95,
    rope=rope_interval,
    figsize=(8, 5),
    textsize=12
)

plt.show()
```

ROPE定义为零，表示两个条件差异为零的区间。可以直接赋值给变量或在代码中设置，通过图表显示ROPE与HDI的重叠情况，帮助判断差异是否具有实际意义。

![img 15](./img%2015.png)

绿色部分表示ROPE区间（-0.05到0.05），蓝色曲线表示后验分布，黑色线表示95%的HDI区间。
我们可以看到 $\beta_1$ 的后验分布主要集中在正值区域，其均值约为 0.15。  

图中的 95% 最高密度区间（HDI）范围为 $[0.059, 0.24]$，且大部分后验分布落在 ROPE 区间 $[-0.05, 0.05]$ 之外，只有 1.9% 的后验分布位于 ROPE 区间内。  

这表明 self 和 other 条件下的反应时间差异在实践上具有显著性，即 $\beta_1$ 的值足够大，可以排除两者在反应时间上的实用等效性。因此，self 和 other 条件之间的差异值得关注。

由于95%的HDI落在ROPE之外，我们有信心认为自我和他人条件之间存在显著差异，$β_1$足够大，可以认为它们不是相等的。这类似于频率主义方法中的显著性推断，只是使用了不同的术语。


## 总结  

* 本节课通过一个简单的线性回归示例，展示了如何使用 PyMC 构建贝叶斯模型，并结合之前的内容对模型结果进行深入分析。  
  * 我们特别关注了先验和后验预测检查的重要性，以评估模型的合理性和预测能力。  
* 此外，我们介绍了如何使用 bambi 来简化线性模型的定义和拟合，使得贝叶斯建模的流程更加便捷。。  
* 最后，我们强调了贝叶斯建模中的关键步骤，从模型构建到结果解释，并认识到 MCMC 方法在近似后验分布中的重要性。  

![Image Name](./img%202.png)

## 使用bambi进行模型定义  
Bambi是一个基于PyMC的包，特别优化了心理学和神经科学领域的使用。它简化了模型构建过程，更符合心理学研究者在R和Python中的使用习惯，例如通过简洁的公式定义自变量和因变量，且支持数据清理（如去除NA值）。尽管如此，课程中建议使用PyMC来定义模型，因为这样有助于从概率编程的角度深入理解模型。对于实际数据分析，Bambi能够帮助用户更高效地构建模型，最终的结果也可以通过ArviZ进行相同的分析。

Bambi 是一个用于贝叶斯统计建模的 Python 包，建立在 PyMC 上。  

```pyhon  
model = bmb.Model('RT_sec ~ Label',  
                  data=df,  
                  dropna=True)  
```

* 它提供了一个更加简化的界面，使得使用贝叶斯统计模型更加容易，特别是在处理线性和广义线性混合模型时。  

* 上面的代码提供了一个示例，我们可以简单的三行代码来完成之前的回归模型的定义。  

* 其中，我们可以通过 lme4 的形式来表达线性关系：`'RT_sec ~ Label'`。~左边是因变量，右边是自变量。  

* 需要注意的是，在bambi中，如果我们不对先验进行定义，它会自动选择一个比较弱(weakly informative)的先验。
我们通过代码示例来说明如何通过 bambi 复现之前的分析：  

1. 首先定义模型
```python
import bambi as bmb

# 定义先验并传入模型中
beta_0 = bmb.Prior("Normal", mu=5, sigma=2)  
beta_1 = bmb.Prior("Normal", mu=0, sigma=1)
sigma = bmb.Prior("Exponential", lam = 0.3)

# 将三个参数的先验定义在字典prior中

priors = {"beta_0": beta_0,
          "beta_1": beta_1,
          "sigma": sigma}

# 定义关系式，传入数据
model = bmb.Model('RT_sec ~ Label',
                  data=df,
                  priors=priors,
                  dropna=True)
# 总结对模型的设置
model
```

2. 拟合模型，使用MCMC方法采样得到后验的近似分布  

* 提示：`model.fit` 基于 `pm.sample` 方法。因此，他们的参数设置是相同可继承的。

```python
# ===========================

# MCMC采样过程

# 注意！！！以下代码可能需要运行几分钟

# ===========================
trace = model.fit(draws=5000,                   # 使用mcmc方法进行采样，draws为采样次数
                  tune=1000,                    # tune为调整采样策略的次数，可以决定这些结果是否要被保留
                  chains=4,
                  random_seed=84735)

```
```
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [sigma, Intercept, Label]

```
```
Sampling 4 chains for 1_000 tune and 5_000 draw iterations (4_000 + 20_000 draws total) took 4 seconds.

```
```python
模型诊断部分的分析和之前直接使用 PyMC 是一致的。
ax = az.plot_trace(trace, figsize=(7,7), compact=False)
plt.tight_layout()
plt.show()
```

![img](./img%2016.png)

```python
az.summary(trace)
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Intercept</th>
      <td>0.712</td>
      <td>0.036</td>
      <td>0.646</td>
      <td>0.781</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29217.0</td>
      <td>15473.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Label</th>
      <td>0.149</td>
      <td>0.046</td>
      <td>0.063</td>
      <td>0.237</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28821.0</td>
      <td>15726.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.232</td>
      <td>0.016</td>
      <td>0.202</td>
      <td>0.262</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29176.0</td>
      <td>15618.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

## 补充材料：为什么使用MCMC是必要的  
最后还有一个补充材料：关于为什么要使用MCMC的而不是用normal或者这种模型计算后验。大家可以去感受一下最后这个公式，如果手动的去求的话，它的复杂程度。
>我们都知道当后验分布的计算过于复杂时，我们应该选用MCMC来近似后验分布  

>但是在这里后验分布究竟有多复杂呢，这里提供了直接的计算(or提供一些复杂的公式让人知难而退)：  

1. 该线性模型存在三个参数值$(\beta_0, \beta_1, \sigma)$  
 * 那么先验概率则为三者pdf的乘积：  

$$  
 f(\beta_0, \beta_1, \sigma) = f(\beta_0) f(\beta_1) f(\sigma)  
$$  

2. 观测到的数据可以用$\vec{y} = (y_1,y_2,...,y_{n})$来表示  
 * 那么似然函数可以表示为：  

$$  
 L(\beta_0, \beta_1, \sigma | \vec{y}) = f(\vec{y}|\beta_0, \beta_1, \sigma) = \prod_{i=1}^{n}f(y_i|\beta_0, \beta_1, \sigma)  
 $$  

3. 后验分布则可以表示为：  

$$  
\begin{split}  
f(\beta_0,\beta_1,\sigma \; | \; \vec{y})  
 & = \frac{\text{prior} \cdot \text{likelihood}}{ \int \text{prior} \cdot \text{likelihood}} \\  
 & = \frac{f(\beta_0) f(\beta_1) f(\sigma) \cdot \left[\prod_{i=1}^{n}f(y_i|\beta_0, \beta_1, \sigma) \right]}  
 {\int\int\int f(\beta_0) f(\beta_1) f(\sigma) \cdot \left[\prod_{i=1}^{n}f(y_i|\beta_0, \beta_1, \sigma) \right] d\beta_0 d\beta_1 d\sigma} \\  
 \end{split}  
 $$  
