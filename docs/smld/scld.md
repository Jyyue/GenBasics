# score matching langevin dynamics

## Not KL divergence, but score matching loss


Generative model usually use KL divergence to match the learnt distribution and model distribution $logp_{data}(x)$ and $logp_\theta(x)$. The score matching model introduce another loss metric: 
 - Score matching loss matches $\nabla_x \log p_{data}(x)$ and $\nabla_x \log p_\theta(x)$
The score matching loss is
 - $\mathcal{L_{sm}} = E_{p_{data}(x)}[\left\|\nabla_x logp_{data}(x)- \nabla_x \log p_\theta(x)\right\|]$
 - Train a score model $S_\theta(x) := -\nabla_x logp_\theta(x)$, optimize $\theta$ by minimizing $\mathcal{L_{sm}} = E_{p_{data}(x)}[\left\|\nabla_x logp_{data}(x)+ S_\theta(x)\right\|]$

But in paper [learn distribution by estimating score](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf), hyvarinen prove that instead of model prosibility distribution itself, we can also learn it from its derivative.


## What is the score of a distribution? 

The score $s(x) = - \nabla \log p(x)$. It is the force of energy landscape of $\log p$.

As in statistical physics, we assign a probability to all possible states (x is continous random variable in phase space). In canonical ensemble, the possibility of finding the state at x is as the following.

$$p(x) = \frac{e^{-\beta E(x)}}{Z}$$
$$Z=\int e^{-\beta E(x)} dx$$

Calculate its log, we get $$log p(x) = -\beta E(x) - log Z$$

This links a distribution with energy function. Bolzmann machine of hopfiled network as an example, define energy function like Ising model, is trained to memory or approximate from data distribution. For more details, see https://courses.physics.illinois.edu/phys498cmp/sp2022/ML/Hopfield.html.

From this point of view, several intuitions from statistical physics emerge:

- The probability distribution \( p(x) \) is proportional to the degeneracy \( \Omega(x) \), and the inverse temperature \( \beta \) is proportional to \( 1/T \). The degeneracy \( \Omega(x) \) itself is proportional to \( \exp(E/T) \).

- Therefore, when there is only a small change in energy \( \Delta E \), the corresponding change in probability is exponential, i.e., \( \exp(\Delta E/T) \). This implies that even if the energy surface is relatively smooth, the probability distribution is highly rugged and sensitive to changes in energy.

- At low temperatures (small \( T \)), or in situations where sampling is local, the probability landscape becomes sharp and peaked. In contrast, at high temperatures, the probability surface is flattened—making all states have nearly similar probabilities.

Moreover, if our phase space is continous, take derivative, we have $$\nabla log p(x) = -\beta \nabla_x E(x) = \beta \vec{F}$$

Thus, three correlated quantity is given. Lets ignore the $\beta$
$$log p\Leftrightarrow  -E$$
$$\nabla log p(x) \Leftrightarrow  F$$

The force is important, because it gives what is needed for dynamic of the system. Imagine those x are coordinates, then force can be used for finite temperature md to sample from it.
> correlation between MCMC and MD? If MC is required to move to nearby neighbors, then it approximates to score.

What is the benefit of force? One application is simulation, with probability, we have various methohs such as MCMC to draw sample from distribution. Now with force, dynamic simulation is also possible. Example is langevin dynamics. This idea corresp to a family of gradient based models.

Score function and energy model
 - Energy model in machine learning: $p(x) = \frac{e^{-\beta E(x)}}{Z}, \quad Z = \int e^{-\beta E(x)} dx$
 - score function is the force field defined by energy function $\nabla_x \log p(x) = -\beta \nabla_x E(x) = \beta \cdot \vec{F}(x)$


## To calculate score, we need noise

Let's look at the score matching loss $\mathcal{L_{sm}} = E_{p_{data}(x)}[\left\|\nabla_x logp_{data}(x)- \nabla_x \log p_\theta(x)\right\|]$. The score of data distribution need to be calculated, but we can't calculate data distribution. If we already have the form of it, we already solve generation problem!

Emperical data has no gradient, however we can train a score model on pertubated data distribution $p_{\sigma}(\tilde{x}) = \int p(x) q_\sigma(\tilde{x}|x) dx$, which trasforms delta distribution to a smooth distribution with gradient. The score of the pertubated data distribution $\mathcal{L_{dsm}} = \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x} \mid x)} \left[ \| s_\theta(\tilde{x}) + \nabla_{\tilde{x}} \log p(x, \tilde{x}) \|^2 \right]$. Vincent proves that $\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x) = \nabla_{\tilde{x}} \log p(x, \tilde{x})$. With this result, $\mathcal{L_{dsm}} = \mathbb{E}_{p(x)} \mathbb{E}_{q_\sigma(\tilde{x} \mid x)} \left[ \left\| s_\theta(\tilde{x}) + \nabla_{\tilde{x}} \log q(\tilde{x} \mid x) \right\|^2 \right]$
**With the form of pertubation, we can write down the $L_{DSM}$, with only $s_\theta$ unkonwn!**

Better score matching with multi scale noise
 - $\mathcal{L_{dsm}} = \mathbb{E}_{t\sim U[0, 1]} \mathbb{E}_{p_t(x)} \mathbb{E}_{q(\tilde{x} \mid x)} \left[ \| s_\theta(\tilde{x}, t) + \nabla_{\tilde{x}} \log p_t(x, \tilde{x}) \|^2 \right]$

This allow us to cover a overall energy landscape, not only the neighbors of the origional manifold. Moreover, we can do annealing sampling， start from large noise then to smaller one.

E.g. Gaussian pertubation kernel $q_{\sigma_t}(\tilde{x}|x) = \mathcal{N}(x, \sigma_t^2 I)$


 - $p_{\sigma_t}(\tilde{x}) = \int p(x) q_{\sigma_t}(\tilde{x}|x) dx = \frac{1}{N} \sum_{i=1}^N \mathcal{N}(x_i, \sigma_t^2 I)$
 - $s(x, t) =  \frac{1}{\sigma(t)^2} (\tilde{x} - x)$
 - $\mathcal{L}_{dsm} = \mathbb{E}_{t\sim U[0, 1]}\mathbb{E}_{p_{\text{data}}(x) q_{\sigma(t)}(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}, t) + \frac{1}{\sigma(t)^2} (\tilde{x} - x) \right\|^2 \right]$

Training algorithm with gaussian kernel
>Pre: choose $q_t(\tilde{x} | x)$, initialize score network $s_\theta(x, t)$

Train Loop
1. Sample data point and noise level $x \sim p_{data}(x), t\sim U[0, 1]$
2. Add noise: get sample $\tilde{x} \sim q_t(\tilde{x} | x)$ by $\tilde{x} = x + \sigma_t \epsilon$, $\epsilon\sim N(0, 1)$
3. Compute $\frac{1}{\sigma(t)^2} (\tilde{x} - x)$
4. Loss = $s_\theta(\tilde{x}, t) + \frac{1}{\sigma(t)^2} (\tilde{x} - x)$
5. Backpropagate and update $\theta$

## Sampling from score with LD

Sampling with langevin dynamics
 - Langevin dynamics provides an MCMC procedure to sample from a distribution $p(\mathbf{x})$ using only its score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. 
 - Specifically, it initializes the chain from an arbitrary prior distribution $\mathbf{x}_0 \sim \pi(\mathbf{x})$, and then iterates the following: $\mathbf{x}_{i+1} \leftarrow \mathbf{x}_i + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \sqrt{2\epsilon} \mathbf{z}_i, \quad i = 0, 1, \cdots, K$

Sampling algorithm
1. sample x from an arbitrary prior distribution $\mathbf{x}_0 \sim \pi(\mathbf{x})$
2. sample $\mathbf{z}_i$ from normal distribution, update by $\mathbf{x}_{i+1} \leftarrow \mathbf{x}_i + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \sqrt{2\epsilon} \mathbf{z}_i$


**[Question]**: Why langevin dynamics convergence to data distribution?
Ref: ?

## reference

1. DDPM
2. [https://yang-song.net/blog/2021/score/](https://yang-song.net/blog/2021/score/)
4.[Vincent's proof](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)
5.![scoreofgaussian](./graph/score%20of%20gaussian.png "score of gaussian")
6. https://physicsofebm.github.io/

<!--
## Get score from finite data points

Score of discrete data sample is hard to get: They are just emperical delta fucntions. cannot take derivative.

Thus, SMLD gives a solution by make delta distribution Gaussion. $p(\widetilde{x}|x) = \int \mathcal{N}(0, \sigma)p(x)dx = \sum\limits_{x\in D} \mathcal{N}(x, \sigma)$



As a result, the training loss is just
$$\frac{1}{2N}\sum\limits_{1}^{N}E_{\epsilon\in \mathcal{N}(0, \sigma^2 \bold{1})}[||s_\theta(\widetilde{x})+\frac{\epsilon}{\sigma}||^2]$$

We can not let S'(x) = -\sigma S(x), and get
$$\frac{1}{N}\sum\limits_{1}^{N}E_{\epsilon\in \mathcal{N}(0, \sigma^2 \bold{1})}[\frac{1}{2\sigma}||s_\theta(\widetilde{x})-\epsilon||^2]$$

## questions to solve

Exact deriviation of loss, all the term from the begining.

## sde score matching

From the denoising score matching + scheduled score matching, we might realize that noise and noise schedule is important. Then comes a question: can we design the noise schedule? -> noise schedule 规定了probability path，即从一个分布到另一个分布的变换路径，其中变换的步数被用时间t标记

The answer is yes. We can construct forward SDE
$$dx = f(x,t)dt + g(t) dw$$
Then the revere process is given by 
$$dx = [f(x,t)-g^2(t)\nabla_x logp_t(x)]dt + g(t) dw$$

Not much we should let the network learn again! Just score at t distribution is enough.(for a reverse sde solver) To make the process reversable, the p_0 and pt should be known. You can just pick random f(x,t) and g(t), then if you could learn score at each timestep t, the reverse process is OK to go!

## probability flow ode

No randomness! Just go!

ODE.



## 

| **Method**                            | **Description**                                               | **Energy Function Usage**                           | **Applications**                                                       |
|---------------------------------------|---------------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------------------|
| **MCMC (Metropolis-Hastings)**        | Proposes moves and accepts/rejects based on energy differences | Energy difference between proposed and current state | Complex distributions, Bayesian inference                             |
| **Gibbs Sampling**                    | Samples each variable conditionally, based on others          | Conditional energy (or likelihood) for each variable | Bayesian inference, high-dimensional models                           |
| **Hamiltonian Monte Carlo (HMC)**     | Uses Hamiltonian dynamics with energy gradients for proposals  | Gradient of the energy (potential energy)          | High-dimensional Bayesian inference, physical simulations            |
| **Langevin Dynamics**                 | Evolves the system using energy gradients and noise            | Gradient of the energy, with added noise            | Statistical mechanics, machine learning (SGLD for Bayesian inference)  |
| **Rejection Sampling**                | Proposes from a simpler distribution and accepts based on energy | Energy determines acceptance probability            | Simpler distributions, Monte Carlo integrals                          |
| **Importance Sampling**               | Samples from a proposal and weights by energy (importance)     | Energy function used to weight samples             | Approximation of intractable distributions                            |
| **Diffusion Models (Score-Based)**    | Generates data through a reverse diffusion process             | Score function (gradient of log-likelihood)         | Generative modeling, deep learning                                  |
| **Fokker-Planck Equations**           | Describes the evolution of the probability distribution        | Energy function as part of the drift term          | Statistical mechanics, physics-based modeling                        |

A conclusion by GPT, has errors and is misleading some way.

Energy model has some draw backs, such as hard to calculate prossibility for large number of neurons.

Methods to solve this including score matching and contrastive divergence.
> 笔者自己学习contrastive divergence时的资料之一
> https://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf

> 能不能用RBM来denoise呢？虽然但是这样和拟合物理模型参数有什么区别
> 有没有觉得连续情况的denoise 和离散情况的flip spin/move很像


### with score, sample

The x moves follows langevin dynamics. 

$$x_{t+\delta t} = x_t + \alpha \nabla_{x_t}log p_{real}(x) + \eta \epsilon$$

$$x_{t+\delta t} = x_t + \alpha s(x) + \eta \epsilon$$

This is the solution of $F=m\frac{d^2x}{dt^2}$, with a brown motion added.

The full sde is:

𝑑𝑥(𝑡)=𝑎𝑥(𝑡)𝑑𝑡+𝑑𝑤(𝑡)

where 𝑤(𝑡)
 is brownian motion. $$W(t) = \sqrt{t} N(0,1)$$

Moreoover there are more to think about **sampling** mathods! 

Imagine a dynimic process with T decresing, we may explore more modes. This is annaling langevin dynamics. There is a deeper explaincation you may find: the distribution you sample from is actrually changing! Though the outcome is the battle between gravity and heat.

Then for sensitive reader, they may realize that distribution changing is not coincident, but can be a common stegrategy. Changing T is not very much specal, since it is just a controllable factor. However, change distribution itself, can be of more freedom.

Diffusion process is another way of change distribution! The score is now for every time step, not just final distribution.

Difference between sampling from LD and diffsion sampling: The former samples several times for one distribution, the latter sample only once, but sample from many different distribution follow a schedule.
-->