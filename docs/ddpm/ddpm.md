# Denoising Diffusion Probabilistic Models

To build our home... why not start with diffusion model?

## the idea of ddpm

it all start with this famous image in paper:

![ddpm过程](./graph/ddpm_process.png "ddpm过程")

Like most generative models, ddpm models distribution via optimizing model parameters($\theta$).由flow model，我们知道可以定义一些列变量的变换，从而由一个已知的随机变量分布变换到数据的随机变量分布。但flow模型面对诸多限制，比如，每一步路径的形式是受限的，并且每一步变换是确定的。

diffusion和flow模型的一个不同在于，diffusion的变量来自一个随机的path, 而flow的变量来自一个确定的path。diffusion只约束变量的概率分布，而不规定它是用什么方式得到的这种概率分布，因此可以有更高的自由度。图示为diffusion模型的markov chain变量变换过程。它的前向过程是markov的，

$q(x_{1:T} | x_0)$ 的联合概率分解为马尔可夫链的乘积形式：
\[
q(x_{1:T} | x_0) = q(x_1 | x_0) \prod_{t=2}^{T} q(x_t | x_{t-1}) = \prod_{t=1}^{T} q(x_t | x_{t-1}).
\]


diffusion模型约定一个已知的前向过程，通过前向过程构造中间状态变量的样本，然后由这些样本学习一个反向markov过程(可以证明这样的过程是存在的），在采样的时候，只需要从prior分布中采样，然后经过反向过程得到复合原始分布的数据。

![random-variable-path](./graph/random_variable_path.png "random-variable-path")

markov chain的具体形式

定义单步扩散的核为
\[q(x_t|x_{t-1}) = K(x_t|x_{t-1};\beta_t)
\]

下面给出高斯核的前向过程（基于 [Sohl-Dickstein et al., 2015]）：
\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}),
\]
其中：
- \(\mathcal{N}\) 表示高斯分布，
- \(\sqrt{1 - \beta_t} x_{t-1}\) 是均值，
- \(\beta_t \mathbf{I}\) 是协方差矩阵（各向同性噪声）。

定义 \( \alpha_t = 1 - \beta_t \)，扩散过程逐步生成 \( X_t \)：
\[
X_t = \sqrt{\alpha_t} X_{t-1} + \sqrt{1 - \alpha_t} Z_t
\]
递推展开后：
\[
X_t = \sqrt{\alpha_t \alpha_{t-1}} X_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} Z_{t-1} + \sqrt{1 - \alpha_t} Z_t 
\]
合并噪声项：
\[
X_t = \sqrt{\alpha_t \alpha_{t-1}} X_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} Z_{t-1:t}
\]
最终直接关联初始状态 \( X_0 \)：
\[
X_t = \sqrt{\bar{\alpha}_t} X_0 + \sqrt{1 - \bar{\alpha}_t} Z_{0:t}, \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i 
\]

因此前向过程
\[
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left( \mathbf{x}_t; \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I} \right)
\]

可以基于$\mathbf{x}_0$的一步采样
\[
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \varepsilon, \quad \varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\]

>对应多元高斯分布形式为
\[\mathcal{N}(x; \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^k \det \Sigma}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)\right).\]
    - \( x \): 随机变量向量，
    - \( \mu \): 均值向量，
    - \( \Sigma \): 协方差矩阵，
    - \( k \): 向量维度。
若 \( X \sim \mathcal{N}(\mu, \Sigma) \)，可通过线性变换从标准高斯分布生成：\[X = \mu + AZ, \quad Z \sim \mathcal{N}(0, I),\]其中 \( A \) 满足 \( \Sigma = AA^T \)。


可以证明[Feller, 1949], 在无限小的变换近似下，反向过程和正向过程具有相同的分布形式。因此对以上高斯kernel，它的反向过程也是高斯的。进一步可以计算得到闭形式

\[
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1}; \tilde{\mu}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}\right)
\]
其中
\[
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
\]

\[
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0
\]

因此我们有理由用模型反向建模马尔可夫链的联合概率：
\[
P_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T P_\theta(x_{t-1} \mid x_t).
\]

\[
P_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\right).
\]


这里，因此，在反向过程sampling的时候，只有x_0是未知的，这意味着我们只需要一个模型预测inverse mu，就可以给出反向过程

## loss

\[
    \log p(x_0) \geq \mathbb{E}_{q} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \right]
\]

对应的变分下界作为损失函数
\[
L_{\text{VLB}} = \mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] \geq -\mathbb{E}_{q(\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0) \right]
\]


将以上概率通过markov chain展开，并使用bayes公式进行一些变形之后可以得到

\[
    L_{VLB} = \mathbb{E}_{q(\boldsymbol{x}_0)} [ D_{\text{KL}}[q(\boldsymbol{x}_T|\boldsymbol{x}_0)||p(\boldsymbol{x}_T)] + \sum_{t=2}^T \mathbb{E}_{q(\boldsymbol{x}_t|\boldsymbol{x}_0)} \left[D_{\text{KL}}[q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)||p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)]\right] - \mathbb{E}_{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} [\log p_\theta(\boldsymbol{x}_0|\boldsymbol{x}_1)]
\]

两个高斯分布的KL是有公式的，写出来


\[
L_t = \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2 \|\Sigma_\theta(x_t, t)\|_2^2} \left\| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \right\|^2 \right]
\\
= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2 \|\Sigma_\theta\|_2^2} \left\| \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \right) - \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) \right\|^2 \right]
\\
= \mathbb{E}_{x_0, \epsilon} \left[ \frac{1}{2 \|\Sigma_\theta\|_2^2} \left\| \frac{1 - \alpha_t}{\sqrt{\alpha_t (1 - \bar{\alpha}_t)}} (\epsilon_t - \epsilon_\theta(x_t, t)) \right\|^2 \right]
\\
= \mathbb{E}_{x_0, \epsilon} \left[ \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar{\alpha}_t) \|\Sigma_\theta\|_2^2} \left\| \epsilon_t - \epsilon_\theta(x_t, t) \right\|^2 \right]
\\
= \mathbb{E}_{x_0, \epsilon} \left[ \frac{(1 - \alpha_t)^2}{2 \alpha_t (1 - \bar{\alpha}_t) \|\Sigma_\theta\|_2^2} \left\| \epsilon_t - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t, t \right) \right\|^2 \right]
\]

在 DDPM 中，常忽略加权系数，直接优化 \( \|\epsilon_t - \epsilon_\theta(x_t, t)\|^2 \)。

## sampling

\[
q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1}; \tilde{\mu}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}\right)
\]

\[
    \tilde{\mu}(\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
\]
## general ddpm

也许你已经观察到了ddpm和score matching的loss的相似性，实际上，后续有工作证明了优化ddpm的loss可以通过score matching进行

\[
\theta^* = \arg \min_{\theta} \sum_{t=1}^T \sigma_t^2 \mathbb{E}_{q(x_0)} \mathbb{E}_{q(x_t|x_0)} \left[ \| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0) \|_2^2 \right]
\]

- \(\theta^*\)：最优模型参数。
- \(s_\theta(x_t, t)\)：扩散模型学习的**得分函数**（score function），用于近似真实数据分布的梯度 \(\nabla_{x_t} \log q(x_t|x_0)\)。
- \(\nabla_{x_t} \log q(x_t|x_0)\)：真实噪声分布的对数梯度（即 Stein 得分）。
- \(\sigma_t^2\)：时间步 \(t\) 的权重系数（通常与噪声调度相关）。
- \(q(x_0)\)：真实数据分布。
- \(q(x_t|x_0)\)：前向扩散过程的条件分布。

## 
### why ddpm is stacked vae


> In sde/ode, you find similar process, that you usually set x_0 be data and x_T be noise.

>The hidden variables in DDPM is heratical. The data distribution $p_\theta(x_0) = \int dx_{1:T}p(x_{0:T})$, where $p_\theta(x_{0:t}):=p_\theta(x_0|x_1, ..., x_T) = p(x_T)\prod\limits_{t>0} p_\theta(x_{t-1}|x_t)$ is markov and $p\theta(x_{t-1}|x_t)$ is Gaussian.



> except p0 is known, all other distribution is not known so far, but as for generative model, we want to generate the originoal distribution from known distribution, and gaussion is usually x_T.

Generally, generate distribution from another is the same as idea in flow model, etc. We have many middle points, which can be understood from involution of variable. (see this blog for details! I happened to have a glance of it the time writing this note.)

Or, one may understand the ddpm process as a stacked vae. That is a model with lots of hidden, with a known forward process, you try to decode from noised data.

> Before going into details of loss, there is something about *variational inference*  you should know... As the forward and backward is just like encoding(known prior) and decoding in VAE, somebody says that DDPM is just like *stacked VAE*.


Look at our conditional probability. Now we want to optimize the loglikelihood of final data, p. The ddpm actrually takes the idea of variational inference, so the loss log p\theta(x_0) as be written to its elbo

$$logp\theta(x_0) \leq - KL[q(x_{1:T}|x_0)||q(x_{1:T})] + E_{z\sim q(x_{1:T}|x_0)}log p_\theta(x|x_{1:T})$$

This condition distribution form is not trivial. In fact, with the forward prior distribution
$$q(x_{1:T}|x_0) := \prod\limits_{t\leq T} q(x_t|x_{t-1})$$
$$p_\theta(x_T) = \int dx_{0:T-1} q(x_{1:T}|x_0)$$

, we can prove that the backward is also Gaussian, with mean and variance conditional on ($x_0$ and $x_T$)

> proof 

Rewrite the loss,
$$logp\theta(x_0) \leq
-KL[p(x_{1:T}|x_0)||q(x_{1:T})] + E_{z\sim q(x_{1:T}|x_0)}log
p_\theta(x|x_{1:T})$$

From the virational point of view, at each step, the steps before (image) could be regarded as data, and steps after could be regarded as hidden(normal noise).

Each step in backward process can be regarded as mini-vae: $x_t$ is hidden z, and $x_{t-1}$ is x. The q is known as the forward process and has no parameters to learn, the backward process is p.

> only half to learn, decode  from complete noise, not bad

Now, just as VAE loss, with $x$ replaced by $x_0$
$$log p(x_0) \geq E_q[log(\frac{p_\theta(x_{0:T})}{q_\phi(x_{1:T}|x_0)})]$$

*here all x_1,   x_T replaced hidden z in vae formula.

This is the start point of the paper(eq3, optimizing the usual variational bound on negative log likelihood. As here we see it is exactly the lower bound with know prior and encoder, but known decoder to learn)

*Maybe that is when we say that a work had no encoder, but only decoder? haha...

However, remember that we applied a markov process, thus do something to the loss...

$\begin{align}
log p(x_0) \geq &E_q[log(\frac{p_\theta(x_{0:T})}{q_\phi(x_{1:T}|x_0)})]\\
=& E_{q}[log(p(x_T) t\geq(leq1) 1\prod p_\theta(x_{t-1}|x_t))-log(\prod q_\theta(x_{t}|x_{t-1}))]\\
=& E_{q}[log(p(x_t)) + \sum log(\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t}|x_{t-1},x_0)})]\\
\end{align}$

q_\theta(x_{t}|x_{t-1}))
with Bayesian rule, we get 
q(x_{t}|x_{t-1}))=q(xt-1|xt)q(xt)
But none of the term is tractable!

How to solve this? By conditioned on x0
q(xt-1|xt’x0)q(xt|x0)
Too complicated!

But in fact the reverse process is tractable when conditioed on x0
>proof 

As a result, we could condition from xt, equation.

\prod P(xt-1|xt,x0)p(x_T|x0)



Question: when conditioned on x_0, when not? The proof is the answer.j 

as paper appendix proven, this target is equal to
![alt text](<graph/截屏2024-11-17 22.01.58.png>)

The prior q is all known gaussian process, and the p is also gaussian process,thus the kl divergence is between two gaussian distribution, thus can be calculated(the result include the variance and mean of the two gaussian distribution).

The network of diffusion predict the mean and vaiance(but not in ddpm, see paper, since the trainig result is not better).
-----
To prove the Gaussian process
-----


## noise schedule and reparameterization trick

We start from eq(?) in ddpm paper. Focus on the second term only.

$$E_q[KL(q(X_T|x_0)||p(X_T))+\sum\limits_{t\geq 2}KL(q(x_{t-1}|x_t)||p_\theta(x_{t-1}|x_t))-log(p_\theta(x_0|x_1))]$$

Where the forward process(prior distribution) is clear.

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta}x_{t-1}; \beta \bold{I})$$

here the notation $\mathcal{N}$ stands for that $x_t = \sqrt{1-\beta}x_{t-1} + \beta \epsilon$
where $\epsilon= \mathcal{N}(0,1)$

$x_t$ could also be wrtitten conditioned on $x_0$, or both $x_{t+1}, x_0$

$$x_t = \sqrt{\overline{\alpha_t}}x_o + (1-\overline\alpha_t)\epsilon$$

where $\overline{\alpha_t} = \prod \alpha_t$, $\alpha_t = 1-\beta_t$.

$$x_{t-1} = \frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_t}}x_t + \frac{\sqrt{\overline{\alpha_{t-1}}}\beta_t}{1-\overline{\alpha_t}} x_0$$

(Alert: there is something wrong with this line, use bayes rule to strictly dirive this line!)

We can also justify the backward is Gaussion from bayes...?

and

$$x_{t-1} = Kx_t + mu \epsilon$$

This is line is also not correct, you should take mean and variance, instead of this variable relation form.

We have the direct relationship between those variables, x_t, x_0, if epsilon t is known. That is why we can recreate the mean from epsilon and x_t instead of x_t and x_0.

Or, once x_t, epsilon or x_0 2/3 is known, the exact mean of backward gaussian is known.

So rewrite mean as K x_t + K2 epsilon_t, insert to KL loss, keep both the epsilon_t form, and the loss is between epsilon.

with all $x_t$ conditioned on $x_0$ actrually.

$$q(x_{t}|x_{t-1}, x_0) = \mathcal{N}(x_{t-1}, \mu(x_t, x_0),\beta_t \bold{I})$$
$$x_{t-1} = \mu(x_t, x_0) + \beta_t \epsilon$$

This is what we know about prior q, which takes half of the KL loss(second term). To get the optimization loss, we need also the deccoder p.

As we see, the backward process is also gaussian, so there are two ways to parameter x_{t-1}

one way is $x_{t-1} = N(\mu_\theta(x_t, t), \sigma_t^2)$

here the variance term use the same as prior conditioned on x_0 we already known(which could be learnable, but not here), and the variance is to be learnt.

Now the epsilon is conditioned on t.
----

reparameterization trick

Here we already applyed the reparameterization trick. Remembre how wo get variable x_t? not directly from dstribution with mean and varianve, but sample noise, then multi-by variance and added by mean. What is the benefits?

Since we are chracter distribution, just think about KL on distribution. the loss with z will becomes not directly from certain parameter, but sample from distribution. Thus the bp is impossible. However, with z expressed explicitly with model parameter, loss contains paras and bp is possible for optimization.

Take KL loss term as example. Imagine we have two gaussian random variable, and want to calculate loss. It is impossible to directly sample the z and then calculate the loss(when assume the sigma is the same). because mu is not parameterd by distribution paras.

z ~ N(mu, sigma I)

z = mu + sigma epsilon

epsilon ~ N(0, 1)

## training process

With loss and parameterizatin, the training loss becomes clear

E_q[||epsilon-epsilon||]

