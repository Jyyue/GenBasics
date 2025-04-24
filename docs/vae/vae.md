# Notes about variational optimization

## ELBO

生成模型的优化目标是KL散度


   \[
   \begin{aligned}
   \log p_\theta(x) 
   &= \log \int p_\theta(x|z) p(z) \, dz \\
   &= \log \int p_\theta(x|z) \frac{p(z)}{q_\phi(z|x)} q_\phi(z|x) \, dz \quad \text{(引入变分分布 $q_\phi(z|x)$)} \\
   &\geq \int q_\phi(z|x) \log \left[ \frac{p(z) p_\theta(x|z)}{q_\phi(z|x)} \right] dz \quad \text{(Jensen不等式)} \\
   &= \mathbb{E}_{q_\phi(z|x)} \left[ -\log \frac{q_\phi(z|x)}{p(z)} + \log p_\theta(x|z) \right] \\
   &= -\text{KL}\big[ q_\phi(z|x) \| p(z) \big] + \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right] \quad \text{(ELBO)}
   \end{aligned}
   \]

   \[
    -\mathbb{E}_{p_{\text{data}(x)}} \log p_\theta(x) 
   \leq \mathbb{E}_{p_{\text{data}(x)}} \left[ \text{KL}\big[ q_\phi(z|x) \| p(z) \big] - \mathbb{E}_{q_\phi(z|x)} \log p_\theta(x|z) \right] := \mathcal{L}_{\text{VAE}} 
   \]

The difference between elbo and true $\log p$ is $KL[q_\phi(z|x)||p(z|x)]$.

## VAE 训练

   \[
   \mathcal{L}_{\text{VAE}} 
   = \mathbb{E}_{p_{\text{data}(x)}} \left[ \text{KL}\big[ q_\phi(z|x) \| p(z) \big] - \mathbb{E}_{q_\phi(z|x)} \log p_\theta(x|z) \right]
   \]

VAE通过同时训练隐变量z条件x的变分分布和x｜z的分布来优化ELBO。q起到了变分的作用：在训练过程中逐渐接近先验的p(z)分布。

当ELBO完全贴近logp时，对应最好的q(z|x)先验应该满足$q_\psi(Z|x) = p(z|x)$。但实际训练中，很难做到这点。这会导致，即便ELBO term已经被优化，loss仍然和true loss有相当的距离。

![alt text](<graph/ELBO.png>)

编码器:
Encoder(x) -> mu(z), sigma(z), models q(z|x)
Decoder(z) -> log p(x), models p(x|z)




* what bound? as the q_\phi get closer to true posterior, the bound becomes tight, and the optimzation target approximation becomes better.
Moreover, ELBO maximize KL between proir and p posterior, and minimize reconstruction loss.

As a result, the q(z|x) \sim p(z|x) \sim p(z)
The former regular encoder and the latter regular decoder.

With this elbo, we can optimize loglikelihood of x and posterior simutaniously!

## calculation of KL term and reconstruction term

KL term between two Gussian distribution can be calculated with the forlmula:



For continous data, the reconstruction term is modeled by gaussian distribution centered at data point, which lead to mse reconstruction loss. (we can't model $\log p(x|z)$ directly for continous case)

## reparameterization trick

If directly sample from q(z|x) and then calculate reconstruction loss, we can't track the gradient on distribution paramter mu.

As a result, instead of sample from the Gaussian distribution directly, we first sample from normal distribution, then construct z = mu + sigma * eps.

## pesudocode


Training

1. get data batch x
2. calc q conditioned on x （gaussian with mean variance）
3. sample eps, z = mu + sigma*eps（reparameterization trick）
4. predict x distribution，这时候默认从z预测到的x是高斯的分布，预测模型得到的是mean（可以有variance）并且假设采样来自本身也是高斯，因此两个高斯做log差得到的是mse loss。看起来就像预测的mse loss，其实还是p（x｜z）概率模型带回去公式和p（x）计算。

Sampling

1. sample z from p（z）采样（因为假设model训练好后prior对齐了）
2. predict log p(x|z) and sample from it(discrete), or use decoder to decode x directly(continous)

## 先验塌缩

kL divergence代表了logp和变分下届之间的间隔，存在一种可能，这个间隔很大（先验cllapes）。
因为loss中促使先验向无条件先验接近，如果编码器很强，那么它可以从完全无条件的先验中产生样本。

## ref

1. https://jmtomczak.github.io/
2. [苏剑林‘s blog](https://www.spaces.ac.cn/archives/5253)




