# score matching Langevin dynamics

## Not KL divergence, but score matching loss


Generative models usually use KL divergence to match the learned distribution $\log p_\theta(x)$ and model distribution $\log p_{data}(x)$. In the paper [learn distribution by estimating score](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf), hyvarinen proves that instead of model probability distribution itself, we can also learn it from its derivative. 

The score matching model introduces another loss metric: Score matching loss matches $\nabla_x \log p_{data}(x)$ and $\nabla_x \log p_\theta(x)$.
 - $\mathcal{L_{sm}} = E_{p_{data}(x)}[\left\|\nabla_x logp_{data}(x)- \nabla_x \log p_\theta(x)\right\|]$

Guided by this loss, we train a score model $S_\theta(x) := -\nabla_x logp_\theta(x)$, optimize $\theta$ by minimizing $\mathcal{L_{sm}} = E_{p_{data}(x)}[\left\|\nabla_x \log p_{data}(x)+ S_\theta(x)\right\|]$


## What is the score of a distribution? 

The score $s(x) = - \nabla \log p(x)$. It is the force of the energy landscape of $\log p$.

As in statistical physics, we assign a probability to all possible states. The possibility of finding the state at x is

\[p(x) = \frac{e^{-\beta E(x)}}{Z}\]
\[Z=\int e^{-\beta E(x)} dx\]

Calculate its log, we get \[\log p(x) = -\beta E(x) - log Z\]

This links a distribution with energy function. 

Different from statistical physics problems, where we usually know the Hamiltonian and find its energy minimum. In generative models, we don't know the energy landscape but know the modes(with relatively low energy). This is like the inverse of statistical physics: we find the Hamiltonian for data distribution, which leads to low energy states observed. Hopfiled network defines energy functions like the Ising model, and trains the parameters to memory data(low energy modes). For more details, see https://courses.physics.illinois.edu/phys498cmp/sp2022/ML/Hopfield.html.


Moreover, if our phase space is continuous, take derivative, we have 
\[\nabla \log p(x) = -\beta \nabla_x E(x) = \beta \vec{F}\]

Thus, three correlated quantities are given. Lets ignore the $\beta$
\[log p\Leftrightarrow  -E\]
\[\nabla log p(x) \Leftrightarrow  F\]

The force is important because it gives what is needed for the dynamic of the system. Imagine that x are coordinates, then force can be used for finite temperature md to sample from it.
> correlation between MCMC and MD? If MC is required to move to nearby neighbors, then it approximates to score.

The force tells the $p(x+\Delta x)/p(x)$, which can be used to construct a continous version of MCMC.(Where we need $p(s_1)/p(s_2)$ for MC)

What is the benefit of force? One application is a simulation, with probability, we have various methods such as MCMC to draw samples from the distribution. Now with force, dynamic simulation is also possible. An example is Langevin dynamics. This idea corresp to a family of gradient-based models.

**Conclusion**:

 - Energy model in machine learning: $p(x) = \frac{e^{-\beta E(x)}}{Z}, \quad Z = \int e^{-\beta E(x)} dx$
 - score function is the force field defined by energy function $\nabla_x \log p(x) = -\beta \nabla_x E(x) = \beta \cdot \vec{F}(x)$



## To calculate the score, we need noise

Let's look at the score-matching loss 
\[\mathcal{L_{sm}} = E_{p_{data}(x)}[\left\|\nabla_x logp_{data}(x)- \nabla_x \log p_\theta(x)\right\|]\]

The score of data distribution needs to be calculated, but we can't calculate data distribution. If we already have the form of it, we already solve the generation problem!

Empirical data has no gradient. However, we can train a scoring model on perturbed data distribution $p_{\sigma}(\tilde{x}) = \int p(x) q_\sigma(\tilde{x}|x) dx$, which transforms delta distribution to a smooth distribution with gradient. The score of the perturbed data distribution \[\mathcal{L_{dsm}} = \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x} \mid x)} \left[ \| s_\theta(\tilde{x}) + \nabla_{\tilde{x}} \log p(x, \tilde{x}) \|^2 \right]\]

Vincent proves that $\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) = \nabla_{\tilde{x}} \log p(x, \tilde{x})$. With this result, \[\mathcal{L_{dsm}} = \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x} \mid x)} \left[ \left\| s_\theta(\tilde{x}) + \nabla_{\tilde{x}} \log q(\tilde{x} \mid x) \right\|^2 \right]\]
**With the form of perturbation, we can write down the $L_{DSM}$, with only $s_\theta$ unknown!**

## Better score matching with multi-scale noise
 - $\mathcal{L_{dsm}} = \mathbb{E}_{t\sim U[0, 1]} \mathbb{E}_{p_t(x)} \mathbb{E}_{q(\tilde{x} \mid x)} \left[ \| s_\theta(\tilde{x}, t) + \nabla_{\tilde{x}} \log p_t(x, \tilde{x}) \|^2 \right]$

![whynoise](<./graph/whynoise.png>)

This allows us to cover an overall energy landscape, not only the neighbors of the original manifold. Moreover, we can do annealing sampling, starting from noisy distribution to explicit distribution.

E.g. Gaussian pertubation kernel $q_{\sigma_t}(\tilde{x}|x) = \mathcal{N}(x, \sigma_t^2 I)$
 - $p_{\sigma_t}(\tilde{x}) = \int p(x) q_{\sigma_t}(\tilde{x}|x) dx = \frac{1}{N} \sum_{i=1}^N \mathcal{N}(x_i, \sigma_t^2 I)$
 - $s(x, t) =  \frac{1}{\sigma(t)^2} (\tilde{x} - x)$
 - $\mathcal{L}_{dsm} = \mathbb{E}_{t\sim U[0, 1]}\mathbb{E}_{p_{\text{data}}(x) q_{\sigma(t)}(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}, t) + \frac{1}{\sigma(t)^2} (\tilde{x} - x) \right\|^2 \right]$

Training algorithm with Gaussian kernel
>Pre: choose $q_t(\tilde{x} | x)$, initialize score network $s_\theta(x, t)$

Train Loop
1. Sample data point and noise level $x \sim p_{data}(x), t\sim U[0, 1]$
2. Add noise: get sample $\tilde{x} \sim q_t(\tilde{x} | x)$ by $\tilde{x} = x + \sigma_t \epsilon$, $\epsilon\sim N(0, 1)$
3. Compute $\frac{1}{\sigma(t)^2} (\tilde{x} - x)$
4. Loss = $s_\theta(\tilde{x}, t) + \frac{1}{\sigma(t)^2} (\tilde{x} - x)$
5. Backpropagate and update $\theta$

## Sampling from score with LD

Sampling with Langevin dynamics
 - Langevin dynamics provides an MCMC procedure to sample from a distribution $p(\mathbf{x})$ using only its score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. 
 - Specifically, it initializes the chain from an arbitrary prior distribution $\mathbf{x}_0 \sim \pi(\mathbf{x})$, and then iterates the following: $\mathbf{x}_{i+1} \leftarrow \mathbf{x}_i + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \sqrt{2\epsilon} \mathbf{z}_i, \quad i = 0, 1, \cdots, K$

![LD](graph/langevin.gif)

Sampling algorithm
1. sample x from an arbitrary prior distribution $\mathbf{x}_0 \sim \pi(\mathbf{x})$
2. sample $\mathbf{z}_i$ from normal distribution, update by $\mathbf{x}_{i+1} \leftarrow \mathbf{x}_i + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \sqrt{2\epsilon} \mathbf{z}_i$


**[Question]**: Why Langevin dynamics convergence to data distribution? What is sde/ode of shocastic process?
Ref: 

## reference

1. [Generative Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2019](https://arxiv.org/pdf/1907.05600)
2. [ScoreMatching-YangSong](https://yang-song.net/blog/2021/score/)
3. [Vincent's proof](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)
4. [ScoreMatching-jmtomczak](https://jmtomczak.github.io/)
5. [Physics of EBM](https://physicsofebm.github.io/)
