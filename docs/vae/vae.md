# Notes about variational optimization

## hidden variables
What is the idea of variational inference? From statistical point of view, the observed variable is associated with a hidden variable z, that is to say, the observation is somehow conditioned on an unseen variable. 

Some hidden factors...

Gene <-> Expression
Customer requirement <-> price
...

Question: 
1. why we need hidden viriable? (like, no hidden with parameter in sde/ode, flow model)
xhat ->  theta -> x
2. Why is hidden viriable conditioned on x?(or you may think of ode/sde with hidden dependent from x)

1. wrong, sde ode has hidden but they don't need to learn.
2. In bayes inference, and hidden has distribution. We take part of them as cause of observation, part as model parameter(none bayes view)
Then you can understand this by simply understand why we need posterior in bayes inference...
>The unknown variables should have a known distribution that is easy to get. For example, in PCA we do a linear transformation to project our variable to a low dimensional space, we can first get a distribution from that space, then transform to get data distribution. (not always true!)

In bayes inference, we need posterior, which is hard to calculate because of the intractable problem. with complexity **o(e^n)** of computing the integral, n is the dimension of hidden. This make the posterior distribution $p_\theta(z|x)=\frac{p_\theta(x|z)p(z)}{p(x)}$ intractable. here px is intractable， because of integral in z space.

p(z|x) = p(z,x)/p(x) = p(x|z)p(z)/p(x|z)p(z)dz
=> p(x) intractable

Same problem happen if one want to optimize log p(x) by **maximizing loglikelihood**.

variational inference gives one solution to the problem: give a series of function to approximaate intractable posterior, and minimize KL divergence.
>Some words: To utilize hidden, we need to sample from it, then use decoder to get x. Not just viriational inference solve the problem, other model like MC can also be used to model hidden and sample from it.

## ELBO
Question: how is optimize z different from optimize theta in probabilistic model?

* notice: here, the prior and posterior is not on parameter space, but variable space.
A: not much difference in bayes(all variables, distribution)

Look at this loss, what we want is p(x|z), instead of calculate it directly, we can introduce a distribution to approximate it.
It parameter the $q_\phi(z|x)$ to approximate $p_\theta(z|x)$. The approximate, means to minimize the KL divergence between the two distribution.

$$KL(q_\phi(z|x)||p_\theta(z|x))$$

Do some formulation things, we can get the ELBO:

$\begin{align}
&KL(q_\phi(z|x)||p_\theta(z|x))\\
=&\int q_\phi(z|x)dz [log(q_\phi(z|x))-log(p_\theta(z|x))]\\
=&E_q[log(q_\phi(z|x))-log(\frac{p_\theta(x|z)p(z)}{p(x)})]\\
=&E_q[log(q_\phi(z|x))-log(p_\theta(x|z))-log(p(z))+log(p(x))]\\
=&E_q[log(q_\phi(z|x))-log(p(z))]-E_q[log(p_\theta(x|z))]+E_q[log(p(x))]\\
=&KL[q_\phi(z|x)||p(z)]-E_q[log(p_\theta(x|z))]+log(p(x))
\end{align}$

This bound could also be written as
$$log p(x) \geq E_q[log(\frac{p_\theta(x,z)}{q_\phi(z|x)})]$$

* 注意积分变量为z，因此最后一项px可以拿出去直接变为0

since KL divergence is always greater than zero, we get the inequality:

$$log(p(x)) \geq -KL[q_\phi(z|x)||p(z)]+E_q[log(p_\theta(x|z))]$$

This is to say, instead of optimizing the $logp_\theta(x)$ directly, we might find another question: optimize its variational lower bond. The lower bond optimization is fully tractable, when a encoder net that encode x to its hidden z, and a decoder net that that predict x from given hidden z, with given prior. **To optimize the network, you may need two step optimization? Plus, the q and prior should be some good form so the first term in the loss could be computed easily.**

优化上式相当于寻找theta使得loglikelihood最大。也就是minus右侧最小（梯度下降）。注意：每given x，可以得到的z都不是确定的，而是condition on x含参分布的采样。

因此实际的loss是外面再套一层pxdata

* what bound? as the q_\phi get closer to true posterior, the bound becomes tight, and the optimzation target approximation becomes better.
Moreover, ELBO maximize KL between proir and p posterior, and minimize reconstruction loss.

As a result, the q(z|x) \sim p(z|x) \sim p(z)
The former regular encoder and the latter regular decoder.

With this elbo, we can optimize loglikelihood of x and posterior simutaniously!

训练的伪代码：

get data batch x
calc q conditioned on x （gaussian with mean variance）
sample z（reparameterization trick）
predict x distribution，这时候默认从z预测到的x是高斯的分布，预测模型得到的是mean（可以有variance）并且假设采样来自本身也是高斯，因此两个高斯做log差得到的是mse loss。看起来就像预测的mse loss，其实还是p（x｜z）概率模型带回去公式和p（x）计算。

kL divergence代表了logp和变分下届之间的间隔，存在一种可能，这个间隔很大（先验cllapes）。

> when you don't like the answer, change the question. (is that the word?)

The benefit of optimizing the variational lower bond is, when the lower bond is optimized, you can get the posterior. 不考虑prior cllapse。

采样的伪代码
从p（z）采样（因为假设model训练好后prior对齐了）
计算x的mean和variance，sample to get data


