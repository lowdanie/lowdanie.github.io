---
layout: post
title: "The Cramer-Rao Lower Bound"
date: 2019-05-07
mathjax: true
---

# Introduction

Many real world problems can be formulated as an attempt to estimate the value of a variable based on a collection of noisy measurements. For example, we could try to estimate the mean and standard deviation of a populations height based on the "measurements" coming from the heights of a few specific people. Another common problem is to estimate our position based on measurements of signals from GPS satellites. 

In both of these cases, the measurements are _noisy_ since they do not relate to the variable in question in a deterministic way. In the case of estimating the height distribution of a population, the measurements we get depend on which people we decide to measure. In the GPS example, the signals coming from the satellites may be corrupted by the atmosphere and buildings in unpredictable ways.

Given a set of measurements, there are many algorithms one could use to produce an estimate of the underlying variable of interest. In the first example, a good method for determining the population mean would be to simply take the average of the measured heights. However, no matter which algorithm is used, the noisy nature of the measurements imposes a fundamental limit on how close our estimate can get to the true value.  The _Cramer-Rao Lower Bound_ (CRLB) is a fairly simple formula for determining this limit based only on the measurement specification without concerning ourselves with which particular estimation algorithm is being used. The remainder of this post will introduce the machinery needed to compute the bound, prove its correctness, and then apply it to a few examples. 

# State Estimation

Before we can describe the CRLB we need to formalize some of the basic ingredients of _state estimation_.
For simplicity, in this post we will assume that we are trying to estimate a fixed one dimensional variable $x \in \mathbb{R}$ whose true value is $x_\mathrm{true}$ using a measurement of the form $Z = f(x, w) \in \mathbb{R}^n$ where $w$ is a random variable.
In other words, the measurement depends on both the value of $x$ and the outcome of a randomly generated $w$. For any given value of $x$, we have the conditional probability distribution $p(Z \vert x)$ over all possible measurement outcomes. Note that $p(Z \vert x)$ is determined both by $f$ and by the distribution of $w$. 

An _estimator_ of $x$ is a function $\hat{x} : \mathbb{R}^n \rightarrow \mathbb{R}$. Given a measurement $Z \in \mathbb{R}^n$, $\hat{x}(Z)$ is our estimate of $x$ based on $Z$.

There are many ways to evaluate the quality of an estimator. One important one is called the _bias_ which is defined as: 

\\[
E[\hat{x}(Z) - x_\mathrm{true}]
\\]

where the expectation is taken over the measurements $Z$ using the density function $p(Z \vert x_\mathrm{true})$. An estimator is called _unbiased_ if its bias is equal to $0$. Another important metric is the _mean squared error_ which is defined as

\\[
E[(\hat{x}(Z) - x_\mathrm{true})^2]
\\]

where the expectation is defined as above.

# The Cramer-Rao Lower Bound

We are now in a position to give a precise statement of the CRLB. Let $\hat{x}(Z): \mathbb{R}^n \rightarrow \mathbb{R}$ be an unbiased estimator of $x$. Then, the mean squared error is bounded from below as follows:

\\[
E[(\hat{x}(Z) - x_\mathrm{true})^2] <= J^{-1}
\\]

where 

\\[
J = -E\left[\frac{\partial^2\ln{p(Z | x)}}{\partial^2 x}\right]\bigg\rvert_{x_\mathrm{true}} = E\left[\left(\frac{\partial\ln{p(Z | x)}}{\partial x}\right)^2\right]\bigg\rvert_{x_\mathrm{true}}
\\]

$J$ is called the _Fischer Information_. The equality of the two expressions for $J$ will be proved below. The key fact here is that $J$ depends only on the measurement distribution, and not on the estimator. Note that in this case $J$ is a scalar, but there is also a more general version for when the state $x$ is a vector, in which case $J$ is a matrix. 

We now turn to the proof of the lower bound. We start by simply writing down the unbiasedness assumption:

\\[
E[\hat{x}(Z) - x] = \int_{-\infty}^{\infty}[\hat{x}(Z) - x]p(Z | x)dZ = 0
\\]

Note that we have replaced $x_\mathrm{true}$ with $x$ in the definition of the bias since the unbiasedness condition holds regardless of the true value of the parameter $x$. Now, comes the key step of the proof. We have an identity involving the mean of $\hat{x}(Z) - x$ and we would like to deduce a bound on the mean of its square $(\hat{x}(Z) - x)^2$. For this we use a standard trick which seems unmotivated the first time you see it but which is quite common in the literature. As a first step, we differentiate the above expression with respect to $x$:


<div style="font-size: 1.3em;">
\begin{align*}
\frac{d}{dx} \int_{-\infty}^{\infty}[\hat{x}(Z) - x]p(Z | x)dZ & =
\int_{-\infty}^{\infty}\frac{\partial}{\partial x}([\hat{x}(Z) - x]p(Z | x))dZ \\
&=  -\int_{-\infty}^{\infty}p(Z | x)dZ + \int_{-\infty}^{\infty}[\hat{x}(Z) -x]\frac{\partial p(Z | x)}{\partial x}dZ \\
& = 0
\end{align*}
</div>

The first integral in the sum is equal to $1$ by the definition of a pdf.
As for the second integral, we use the following identity to recover $p(Z \vert x)$:

\begin{equation}\label{partial-to-log}
\frac{\partial p(Z | x)}{\partial x}  = \frac{\partial \ln{p(Z | x)}}{\partial x}p(Z | x)
\end{equation}

We can now use the integral expression to conclude that:

\\[
\int_{-\infty}^{\infty}[\hat{x}(Z) - x]\frac{\partial \ln{p(Z \vert x)}}{\partial x}p(Z | x)dZ = 1
\\]


which can be rewritten as:

\begin{equation}\label{integral-product}
\int_{-\infty}^{\infty}([\hat{x}(Z) - x]\sqrt{p(Z | x)})\left(\frac{\partial \ln{p(Z | x)}}{\partial x}\sqrt{p(Z | x)}\right)dZ = 1
\end{equation}

Finally, we can now use the _Schwartz Inequality_ to bound $\int_{-\infty}^{\infty}([\hat{x}(Z) - x]^2 p(Z \vert  x))$ in terms of $\int_{-\infty}^{\infty}\left(\frac{\partial \ln{p(Z \vert x)}}{\partial x}\right)^2 p(Z \vert x)dZ$. Recall that the Schwartz inequality states that for a pair of function $f$ and $g$:

\\[
\vert\langle f, g \rangle \vert := \int_{-\infty}^{\infty} f(z)g(z)dz \leq \Vert f \Vert \cdot \Vert g \Vert
\\]

where $\Vert f \Vert := \sqrt{\langle f, f \rangle} = \left(\int_{-\infty}^{\infty} f^2(z)dz\right)^{1/2}$.

Applying this to \ref{integral-product} we get

\\[
\left(\int_{-\infty}^{\infty}[\hat{x}(Z) - x]^2 p(Z | x)dZ \right)^{1/2}
\left(\int_{-\infty}^{\infty}\left(\frac{\partial \ln{p(Z | x)}}{\partial x}\right)^2 p(Z | x)dZ \right)^{1/2} \geq 1
\\]

which can be written as

\\[
E[(\hat{x}(Z) - x)^2] \geq E\left[ \left(\frac{\partial \ln{p(Z | x)}}{\partial x}\right)^2 \right]^{-1}
\\]

Evaluating this inequality at $x = x_\mathrm{true}$ gives us the desired result.

It remains to prove that the two expressions we gave for $J$ agree. Namely, that

\\[
-E\left[\frac{\partial^2\ln{p(Z | x)}}{\partial^2 x}\right]\bigg\rvert_{x_\mathrm{true}} = E\left[\left(\frac{\partial\ln{p(Z | x)}}{\partial x}\right)^2\right]\bigg\rvert_{x_\mathrm{true}}
\\]

This involves essentially the same trick as the proof of the bound itself. We start with the general pdf equality

\\[
\int_{-\infty}^{\infty} p(Z | x)dZ = 1
\\]

After differentiating with respect to $x$ and applying \ref{partial-to-log} we obtain

\\[
\int_{-\infty}^{\infty}\frac{\partial \ln{p(Z | x)}}{\partial x}p(Z | x)dZ = 0
\\]

Differentiating again with respect to $x$ using the chain rule gives the desired equality.

# Example: Estimating Distance

We now apply the CRLB to a simplified robotics setting. Suppose you have a range finder that measures the distance to the nearest point in front of you. For example, you may put such a sensor in the front of a robot to determine how far it can go before hitting something. Furthermore, we will assume that the distance $z$ returned by the sensor is only an approximation of the true distance $x_\mathrm{true}$. Specifically, each time we make a measurement we get

\\[
z = x_\mathrm{true} + w
\\]

where $w$ is a random variable with distribution $\mathcal{N}(0, \sigma)$. In this section we will compute a lower bound on the error we should expect to have (i.e the mean squared error) if we try to produce an unbiased estimate of $x_\mathrm{true}$ using $k$ range measurements.

Formally, our state $x$ represents the distance to the nearest obstacle, and for each state we have an observation $Z \in \mathbb{R}^k$ consisting of the $k$ measurements. It is easy to see that $Z$ is a multivariate gaussian with the density function

\\[
p(Z | x) = \frac{1}{(\sqrt{2\pi\sigma^2})^k} e^{-\frac{1}{2\sigma^2}\sum_{i=1}^{i=k}(Z_i - x)^2}
\\]

Therefore, the Fischer information $J$ is given by

<div style="font-size: 1.3em;">
\begin{align*}
-E\left[\frac{\partial^2\ln{p(Z | x)}}{\partial^2 x} \right] \bigg\rvert_{x_\mathrm{true}} &=
-E\left[\frac{\partial^2}{\partial^2 x} \left(
-\ln{(\sqrt{2\pi\sigma^2})^k} - \frac{1}{2\sigma^2}\sum_{i=1}^{i=k}(Z_i - x)^2
\right)\right] \bigg\rvert_{x_\mathrm{true}} \\
&= \frac{k}{\sigma^2}
\end{align*}
</div>

By the CRLB we have:

\\[
E[(\hat{x}(Z) - x_\mathrm{true})^2] \geq \frac{\sigma^2}{k}
\\]

In particular, we can easily compute how many measurements are required in order to reach a specified precision, even before we decide on an estimation algorithm.
