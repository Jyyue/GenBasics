# score matching the easy way

The generative model usually minimize $KL[p_{data}(x)||p_\theta(x)] = \int dx p(x)log[\frac{p(x)}{p_\theta(x)}]$

However, instead of matching the p directly, we can also match the direvative of logp_\data(x). It is proven in paper[] that the two loss

E_{p_{data}}[\nalda p_{data}(x) - \nalda p_\theta(x)]

Here, for some unknown reason(maybe added here?) Comp define the score of a distribution as s(x) = -\nalda p_\theta(x).

> However, from physics point of view, 
> As in statistical physics, we assign a probability to all possible states (x is continous random variable in phase space). In canonical ensemble, the possibility of finding the state at x is as the following.

$$p(x) = \frac{e^{-\beta E(x)}}{Z}$$
$$Z=\int e^{-\beta E(x)} dx$$

Calculate its log, we get $$log p(x) = -\beta E(x) - log Z$$

This links a distribution with energy function.

Moreover, if our phase space is continous, take derivative, we have $$\nabla log p(x) = -\beta \nabla_x E(x) = \beta \vec{F}$$

The score is just grad of - E(or -logp = E) is Force.

## how to calculated the loss

It is not difficult to model s_\theta by NN. However, we don't know the grad of data distribution.

Score of discrete data sample is hard to get: They are just emperical delta fucntions. cannot take derivative.



denoising score matching prove that, 

Thus, SMLD gives a solution by make pertubation to data we have by a distribution near delta function. $p(\widetilde{x}|x) = \int \mathcal{N}(0, \sigma)p(x)dx = \sum\limits_{x\in D} \mathcal{N}(x, \sigma)$

<修改，这里N分布应该是任何一个noiser分布pertubation，>
然后证明score matching loss 等价于denosiing noise！

## pertubation of different level

E_tE_p_\sigma(t)
<直接写出加了t的公式>