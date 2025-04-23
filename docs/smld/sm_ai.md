# Score Matching the Easy Way — with Denoising

## 1. 目标函数：从 KL 到 Score Matching

传统的生成模型通常通过最小化 **KL散度** 来逼近数据分布：

\[
\text{KL}(p_{\text{data}}(x) \| p_\theta(x)) = \int dx \, p_{\text{data}}(x) \log \left[ \frac{p_{\text{data}}(x)}{p_\theta(x)} \right]
\]

但我们并不一定需要拟合概率密度本身，可以拟合其导数 —— 即 **score function**：

\[
s_\theta(x) := \nabla_x \log p_\theta(x)
\]

理想情况下，我们希望：

\[
\mathbb{E}_{p_{\text{data}}} \left[ \| \nabla_x \log p_{\text{data}}(x) - \nabla_x \log p_\theta(x) \|^2 \right]
\]

然而，**$\nabla_x \log p_{\text{data}}(x)$ 对我们来说是未知的**，因为经验分布只是 delta 函数构成的集合，不可导。

> 一点物理视角

在统计物理中，一个系统状态 $x$ 的概率由 **Gibbs 分布** 给出：

\[
p(x) = \frac{e^{-\beta E(x)}}{Z}, \quad Z = \int e^{-\beta E(x)} dx
\]

取对数得到：

\[
\log p(x) = -\beta E(x) - \log Z
\]

对其取梯度：

\[
\nabla_x \log p(x) = -\beta \nabla_x E(x) = \beta \cdot \vec{F}(x)
\]

所以，score function 在连续系统中等价于 **力场**（即能量梯度）！这启发我们可以用能量建模分布。


## 2. Denoising Score Matching: 解决不可导的问题

由于经验分布不可导，**Score Matching Loss** 无法直接计算。**Denoising Score Matching**（Vincent, 2011）提供了一个优雅的解法：

### Step 1: 添加噪声扰动

对每个样本 $x$ 加噪声（例如高斯）：

\[
\tilde{x} \sim q_\sigma(\tilde{x}|x) = \mathcal{N}(x, \sigma^2 I)
\]

则联合分布为：

\[
p_{\sigma}(\tilde{x}) = \int p(x) q_\sigma(\tilde{x}|x) dx \approx \frac{1}{N} \sum_{i=1}^N \mathcal{N}(x_i, \sigma^2 I)
\]

这将经验分布从 delta 函数“模糊”为一个平滑分布。

\[
\mathbb{E}_{p(x)} \mathbb{E}_{q(\tilde{x} \mid x)} \left[ \left\| s_\theta(\tilde{x}) + \nabla_{\tilde{x}} \log q(\tilde{x} \mid x) \right\|^2 \right]
\]

这个等价于：

\[
\mathbb{E}_{p(x)} \mathbb{E}_{q(\tilde{x} \mid x)} \left[ \| \text{score}(\tilde{x}) + \nabla_{\tilde{x}} \log q(\tilde{x} \mid x) = \nabla_{\tilde{x}} \log p(x, \tilde{x}) \|^2 \right]
\]


### Step 2: 换一种 loss 表达

Vincent 证明了，在这种情况下，score matching loss 等价于以下 Denoising Loss：

\[
\boxed{
\nabla_{\tilde{x}} \log q(\tilde{x} \mid x) = \nabla_{\tilde{x}} \log p(x, \tilde{x})
}
\]
从而

\[
\mathbb{E}_{p_{\text{data}}(x) q_\sigma(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}) + \frac{1}{\sigma^2} (\tilde{x} - x) \right\|^2 \right]
\]

> 换句话说，**模型只需要预测从 $\tilde{x}$ 回到原始 $x$ 的方向和幅度**。

这个损失只依赖于数据点 $x$ 和其噪声扰动 $\tilde{x}$，易于采样和优化。

---

## 4. 多层次扰动（multi-scale noise）

为了让模型在不同噪声强度下都能学习有效的 score，**Sliced Marginal Likelihood Diffusion**（SMLD）方法引入不同的噪声水平：

\[
\sigma \sim p(\sigma), \quad \tilde{x} \sim \mathcal{N}(x, \sigma^2 I)
\]

损失函数变为：

\[
\mathbb{E}_{\sigma \sim p(\sigma)} \, \mathbb{E}_{x \sim p_{\text{data}}} \, \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma^2 I)} \left[ \left\| s_\theta(\tilde{x}, \sigma) + \frac{1}{\sigma^2} (\tilde{x} - x) \right\|^2 \right]
\]

在实现中，$\sigma$ 通常被离散化为一系列噪声等级 $\{\sigma_t\}$，构成一个“噪声调度表”。

---

## 5. 总结

| 方法 | 核心思想 | 优势 |
|------|----------|------|
| Score Matching | 模拟 $\nabla \log p(x)$ | 不需归一化常数 $Z$ |
| Denoising Score Matching | 用噪声扰动平滑数据，从 $\tilde{x}$ 预测回 $x$ | 避免估计 $\nabla \log p_{\text{data}}(x)$ |
| 多尺度 DSM | 引入多个噪声等级，增强泛化能力 | 支持 diffusion 模型等 |







非常棒的问题！我们现在来更精确地写出 **denoising score matching** 中的“noise-score”，即：

\[
\nabla_{\tilde{x}} \log q(\tilde{x} \mid x)
\]

但你问的是——**如何用联合分布 \( p(\tilde{x}, x) \) 来表达它**。这个确实是 score matching 理论的关键之一。

---

## 📌 目标：表达 noise-score 为联合分布的条件导数

我们想找：

\[
\nabla_{\tilde{x}} \log q(\tilde{x} \mid x)
\quad \text{用} \quad p(x, \tilde{x}) = p(x) q(\tilde{x} \mid x)
\]

---

## ✅ 解法：

从贝叶斯公式我们知道：

\[
q(\tilde{x} \mid x) = \frac{p(x, \tilde{x})}{p(x)}
\Rightarrow \log q(\tilde{x} \mid x) = \log p(x, \tilde{x}) - \log p(x)
\]

对 \(\tilde{x}\) 取梯度：

\[
\nabla_{\tilde{x}} \log q(\tilde{x} \mid x) = \nabla_{\tilde{x}} \log p(x, \tilde{x})
\]

（因为 \( \log p(x) \) 对 \( \tilde{x} \) 没有依赖）

---

## ✨ 总结结论：

对于任何扰动分布 \( q(\tilde{x} \mid x) \)，只要你能定义联合分布 \( p(x, \tilde{x}) \)，有：



也就是说，“noise-score”本质上是联合分布对 \(\tilde{x}\) 的 log-gradient。

---

## 🚀 应用到 DSM Loss：

Denoising Score Matching 的 loss 可以写成：

\[
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{p(x)} \mathbb{E}_{q(\tilde{x} \mid x)} 
\left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(x, \tilde{x}) \right\|^2 \right]
\]

---

## 🧠 一点物理直觉（Bonus）

如果你把 \( -\log p(x, \tilde{x}) \) 理解为一个 joint energy，就有：

\[
s_\theta(\tilde{x}) \approx \nabla_{\tilde{x}} \log p(x, \tilde{x})
= - \nabla_{\tilde{x}} E(x, \tilde{x}) \propto \text{“noise force”}
\]

---

如果你想要我给出具体例子，比如：
- Laplace 噪声下的 \(\nabla_{\tilde{x}} \log p(x, \tilde{x})\)
- 或者高斯混合扰动的联合分布和导数

我可以马上写出来！需要吗？
