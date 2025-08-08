---
layout: post
title: "Electron Integrals"
date: 2025-07-26
mathjax: true
utterance-issue: 13
---

# Gaussian Type Orbitals


We will denote the Cartesian coordinates of a point $\mathbf{A}\in\mathbb{R}^3$ by:

$$
\mathbf{A} = (A_x, A_y, A_z)
$$

Similarly, we'll use [multi-index notation](https://en.wikipedia.org/wiki/Multi-index_notation) 
to denote triples of indices with boldface greek letters:

$$
\boldsymbol{\alpha} = (\alpha_x,\alpha_y,\alpha_z)\in \mathbb{N}^3$:
$$

> **Definition (Primitive Cartesian Gaussian).** Let $\boldsymbol{\alpha}\in\mathbb{N}$ be indices,
> $\mathbf{A}\in\mathbb{R}^3$ and $a\in\mathbb{R}\_>$ a positive real number.
> The primitive Cartesian Gaussian with _Cartesian quantum numbers_ $\boldsymbol{\alpha}$,
> _center_ $\mathbf{A}$ and _exponent_ $a$ is defined at position $\mathbf{r}=(x,y,z)\in\mathbb{R}^3$
> by:
>
> $$
> G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A}) := (x - A_x)^{\alpha_x}(y - A_y)^{\alpha_y}(z - A_z)^{\alpha_z} e^{-a ||\mathbf{r} - \mathbf{A}||^2}
> $$

An important property of Cartesian Gaussians is that they can be factorized along the Cartesian 
coordinates into one dimensional Gaussians

$$
G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A}) = G_{\alpha_x}(x, a, A_x)G_{\alpha_y}(y, a, A_y)G_{\alpha_z}(z, a, A_z)
$$

Where the one dimensional Gaussian with quantum number $\alpha\in\mathbb{N}$, 
center $A\in\mathbb{R}$ and exponent $a\in\mathbb{R}$ is defined as:

$$
G_\alpha(x, a, A) := (x - A)^\alpha e^{-a(x - A)^2}
$$

A classic fact about Gaussians is:

$$
\int_\mathbb{R}G_0(x, a, A)dx = \sqrt{\frac{\pi}{a}}
$$

which does not depend on the center $A$. It follows from the above factorization that:

{: #clm:gaussian-integral }
> **Claim (Gaussian Integral).**
>
> $$
> \int_\mathbb{R}G_\mathbf{0}(\mathbf{r}, a, \mathbf{A}) = \left(\frac{\pi}{a}\right)^{3/2}
> $$

# Gaussian Overlaps

> **Definition (Cartesian Gaussian Overlap)**. The _overlap_ of two Cartesian
> Gaussians $G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A})$ and
> $G_\boldsymbol{\beta}(\mathbf{r}, b, \mathbf{B})$
> is defined by:
>
> $$
> \Omega_{\boldsymbol{\alpha}\boldsymbol{\beta}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B}) :=
> G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A})G_\boldsymbol{\beta}(\mathbf{r}, b, \mathbf{B})
> $$

Similarly to the Cartesian Gaussians, the overlaps also factorize along the
Cartesian coordinates into a product of one-dimensional overlaps:

$$
\Omega_{\boldsymbol{\alpha}\boldsymbol{\beta}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B}) =
\Omega_{\alpha_x\beta_x}(x, a, b, A_x, B_x)
\Omega_{\alpha_y\beta_y}(y, a, b, A_y, B_y)
\Omega_{\alpha_z\beta_z}(z, a, b, A_z, B_z)
$$

where the one-dimensional overlap is defined as:

$$
\Omega_{\alpha\beta}(x, a, b, A, B) = G_\alpha(x, a, A)G_\beta(x, b, B)
$$

One key reason that electron integrals of Cartesian Gaussians are easy to evaluate is that the
product of two Gaussians is another Gaussian:

{: #clm:one-dimensional-gaussian-product-rule }
> **Claim (One-Dimensional Gaussian Product Rule).** Let $A,B\in\mathbb{R}$ be two centers and
> $a,b\in\mathbb{R}$ be two exponents. Then:
>
> $$
> G(x, a, A)G(x, b, B) = K(a,b,A,B)G(x, p, P)
> $$
>
> Where new exponent $p\in\mathbb{R}$ and center $P\in\mathbb{R}$ are defined by:
>
> $$
> \begin{align*}
> p &= a + b \\
> P &= \frac{aA + bB}{p}
> \end{align*}
> $$
> 
> and the pre-factor $K(a,b,A,B)$ is defined by:
>
> $$
> K(a,b,A,B) = e^{-\frac{ab}{a + b} (A - B)^2}
> $$

According to the product rule, the one-dimensional Cartesian Gaussian Overlap is
given by:

$$
\Omega_{\alpha\beta}(x, a, b, A, B) = K(a,b,A,B)(x-A)^\alpha(x-B)^\beta G(x, p, P)
$$

The three-dimensional Cartesian Gaussian product rule follows immediately from
the [one dimensional](#clm:one-dimensional-gaussian-product-rule) version and the
factorization XXX

{: #clm:gaussian-product-rule }
> **Claim (Gaussian Product Rule).** Let $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$ be two centers and
> $a,b\in\mathbb{R}$ be two exponents. Then:
>
> $$
> G(\mathbf{r}, a, \mathbf{A})G(\mathbf{r}, b, \mathbf{B}) 
> = \mathbf{K}(a,b,\mathbf{A},\mathbf{B}) G(\mathbf{r}, p, \mathbf{P})
> $$
>
> Where the _total exponent_ $p$ and _reduced exponent_ $\mu$ are defined by:
>
> $$
> \begin{align*}
> p &= a + b \\
> \mu &= \frac{ab}{a + b}
> \end{align*}
> $$
> 
> The new center $\mathbf{P}\in\mathbb{R}^3$ is defined by:
>
> $$
> \mathbf{P} = \frac{1}{p}(a\mathbf{A} + b\mathbf{B})
> $$
>
> and the pre-factor $K(a,b,\mathbf{A},\mathbf{B})$ is defined by:
>
> $$
> K(a,b,\mathbf{A},\mathbf{B}) = e^{-\mu ||\mathbf{A} - \mathbf{B}||^2}
> $$

Here are some simple properties of one dimensional overlaps:

{: #clm:overlap-properties }
> **Claim (Overlap Properties).** Let $\Omega_i$ be the one dimensional overlap
> defined as
>
> $$
> \begin{align*}
> \Omega_i :&= \Omega_{i,0}(\mathbf{r}, a, b, A, B) \\
> &= (x-A)^ie^{-a(x-A)^2}e^{-b(x-B)^2}
> \end{align*}
> $$
>
> Then:
>
> 1. $(x - A)\Omega_i = \Omega_{i+1}$
> 2. $\frac{\partial}{\partial A}\Omega_i = 2a\Omega_{i+1} - i\Omega_{i-1}$
> 3. $\frac{\partial}{\partial B}\Omega_i = 2b\Omega_{i+1} + 2b(A-B) \Omega_i$

{: #def:overlap-transform }
> **Definition (Overlap Transform).**
> Let $a,b\in\mathbb{R}$ be positive real numbers and let $p=a+b$.
> The overlap transform is the linear transformation 
> $T_{ab}:\mathbb{R}^2\rightarrow\mathbb{R}^2$
> defined by the matrix:
>
> $$
> T_{ab} =
> \left[\begin{matrix} \frac{a}{p} & \frac{b}{p} \\ 1 & -1 \end{matrix}\right]
> $$

{: #def:overlap-transform-inverse }
> **Lemma (Overlap Transform Inverse)**
> Let $a,b\in\mathbb{R}$ be positive real numbers. Then, the overlap transform 
> $T_{ab}$ is invertible with the inverse:
>
> $$
> T_{ab}^{-1} =
> \left[\begin{matrix} 1 & \frac{b}{p} \\ 1 & -\frac{a}{p} \end{matrix}\right]
> $$

{: #def:overlap-transform-transform }
> **Claim (Overlap Transform Derivatives).** 
> Let $a,b\in\mathbb{R}$ be positive real numbers and let $p=a+b$. Consider the change of
> coordinates from $(A,B)\in\mathbb{R}^2$ to $(P,Q)\in\mathbb{R}^2$ defined by
> the overlap transform $T_{ab}$:
>
> $$
> \begin{bmatrix}P \\ Q \end{bmatrix} = T_{ab} \begin{bmatrix}A \\ B \end{bmatrix}
> = \begin{bmatrix}\frac{a}{p}A + \frac{b}{p}B \\ A - B \end{bmatrix}
> $$
>
> Let $f:\mathbb{R}^2\rightarrow\mathbb{R}$ be a function defined in coordinates
> $(A, B)$. We can use $T_{ab}^{-1}$ to define $f$ in terms of the coordinates $(P,Q)$:
>
> $$
> f(P, Q) := f(T_{ab}^{-1}(A, B))
> $$
>
> Then, the partial derivatives of $f$ in the $(P, Q)$ coordinates are:
>
> $$
> \begin{align*}
> \frac{\partial}{\partial P}f &= \frac{\partial}{\partial A}f + \frac{\partial}{\partial B}f \\
> \frac{\partial}{\partial Q}f &= \frac{b}{p}\frac{\partial}{\partial A}f - \frac{a}{p}\frac{\partial}{\partial A}B \\
> \end{align*}
> $$

# Hermite Gaussians

The Cartesian Gaussian $G_i(x, a, A) = (x - A)^ie^{-a(x-A)^2}$ is the product of
a degree $i$ polynomial in $x$ and an exponential function. Hermite Gaussians are
another way to generate products of polynomials and exponentials:

> **Definition (Hermite Gaussian)** Let $t\in\mathbb{N}$ be a non-negative integer, $a\in\mathbb{R}_{>}$
> a positive real number and $A\in\mathbb{R}$. 
> The Hermite Gaussian with index $t$, exponent $a$ and center $A$ is defined by:
>
> $$
> \Lambda_t(x, a, A) := \frac{\partial^t}{\partial A}e^{-a(x-A)^2}
> $$

> **Lemma (Differentiation Product Bracket).** For every positive
> $t\in\mathbb{Z}$:
>
> $$
> [\frac{\partial^t}{\partial x^t}, x] = t \frac{\partial^{t-1}}{\partial x^{t-1}}
> $$

> **Claim (Hermite Gaussian Properties).** Let $t\in\mathbb{N}$ be a non-negative integer, $p\in\mathbb{R}_{>}$
> a positive real number and $P\in\mathbb{R}$. Define:
>
> $$
> \Lambda_t := \Lambda_t(x, p, P)
> $$
>
> Then:
>
> 1. $(x - P)\Lambda_t = \frac{1}{2p}\Lambda_{t+1} + t\Lambda_{t-1}$
> 2. $\frac{\partial}{\partial P}\Lambda_t = \Lambda_{t+1}$

> **Definition (Cartesian Overlap To Hermite).**
> Let $i\in\mathbb{Z}$ be a non-negative
> integer, $a,b\in\mathbb{R}$ positive real numbers and $A,B\in\mathbb{R}$.
> Then, the coefficients $E^i_t(a, b, A, B)\in\mathbb{R}$ are defined to satisfy:
>
> $$
> \Omega_i(x, a, b, A, B) = \sum_{t=0}^i E^i_t(a, b, A, B)\Lambda_t(x, p, P)
> $$
>
> Where, as above:
>
> $$
> \begin{align*}
> p &= a + b \\
> P &= \frac{a}{p}A + \frac{a}{p}B
> \end{align*}
> $$
>
> Furthermore, we define $E^i_t(a, b, A, B) = 0$ for all $t < 0$ or $ t > i$.

> **Claim (Cartesian Overlap To Hermite Recurrence).**
> Let $i\in\mathbb{Z}$ be a non-negative integer, 
> $a,b\in\mathbb{R}$ positive real numbers and $A,B\in\mathbb{R}$. Let $p=a+b$
> and $P=\frac{a}{p}A + \frac{b}{p}B$. Define:
>
> $$
> E^i_t := E^i_t(a, b, A, B)
> $$
>
> Then:
>
> 1. $$
>    E^0_0 = K(a, b, A, B)
>    $$
>
> 1. $$
>    E^{i+1}_t = \frac{1}{2p}E^i_{t-1} + (P-A)E^i_t + \frac{i}{2p}E^{i-1}_t
>    $$
>
> 1. $$
>    E^i_{t+1} = \frac{i}{2p(t+1)}E^{i-1}_t
>    $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">
The base case of $E^0_0$ follows immediately from the
[Gaussian Product Rule](#clm:one-dimensional-gaussian-product-rule).

To prove the recurrence relations, let's define:

$$
\begin{align*}
\Omega_i &:= \Omega_i(x, a, b, A, B) \\
\Lambda_t &:= \Lambda_t(x, p, P)
\end{align*}
$$

By the definition of $E^i_t$:

$$
\begin{equation}\label{eq:cartesian-to-hermite}
\Omega_i = \sum_{t=0}^i E^i_t\Lambda_t
\end{equation}
$$

First we'll multiply both side of equation \ref{eq:cartesian-to-hermite} by $(x-A)$.

Starting with the left side, by claim [Overlap Properties](#clm:overlap-properties)
and the definition of $E^i_t$:

$$
\begin{equation}\label{eq:overlap-by-x-a}
\begin{aligned}
(x-A)\Omega_i &= \Omega_{i+1} \\
&= \sum_{t=0}^{i+1} E^i_t\Lambda_t
\end{aligned}
\end{equation}
$$

For the right hand side, by claim 
[Hermite Gaussian Properties](#clm:hermite-gaussian-properties)
and the identity $(x-A) = (x-P) + (P-A)$:

$$
(x-A)\Lambda_t = \frac{1}{2p}\Lambda_{t+1} + (P-A)\Lambda_t +t\Lambda_{t-1}
$$

Inserting this into the right hand side of \ref{eq:cartesian-to-hermite} gives:

$$
\begin{equation}\label{eq:hermite-by-x-a}
\begin{aligned}
(x-A)\sum_{t=0}^i E^i_t\Lambda_t &= \sum_{t=0}^i E^i_t(x-A)\Lambda_t \\
&= \sum_{t=0}^i \frac{1}{2p}E^i_t\Lambda_{t+1} + 
   \sum_{t=0}^i (P-A)E^i_t\Lambda_t +
   \sum_{t=0}^i tE^i_t\Lambda_{t-1} \\
&= \sum_{t=0}^{i+1} \frac{1}{2p}E^i_{t-1}\Lambda_t + 
   \sum_{t=0}^{i+1} (P-A)E^i_t\Lambda_t +
   \sum_{t=0}^{i+1} (t+1)E^i_{t+1}\Lambda_t \\
&= \sum_{t=0}^{i+1} \left(\frac{1}{2p}E^i_{t-1} + (P-A)E^i_t + (t+1)E^i_{t+1}\right)\Lambda_t
\end{aligned}
\end{equation}
$$

Note that to get from the second line to the third line we reindex $t$ and used the
fact that by definition $E^i_t=0$ for $t<0$ and $t>i$.

Equating \ref{eq:overlap-by-x-a} and \ref{eq:hermite-by-x-a} give the following
recurrence:

$$
\begin{equation}\label{eq:cartesian-to-hermite-1}
E^{i+1}_t = \frac{1}{2p}E^i_{t-1} + (P-A)E^i_t + (t+1)E^i_{t+1}
\end{equation}
$$

Next we'll differentiate both sides of \ref{eq:cartesian-to-hermite} by $P$ using
claim [Overlap Transform Derivatives](#clm:overlap-transform-derivatives) 

Starting on the left side, note that by claim
[Overlap Transform Derivatives](#clm:overlap-transform-derivatives):

$$
\frac{\partial}{\partial P} = \frac{\partial}{\partial A} + 
\frac{\partial}{\partial B} 
$$

Therefore, by claim [Overlap Properties](#clm:overlap-properties):

$$
\begin{align}
\frac{\partial}{\partial P}\Omega_i &= 
2a\Omega_{i+1} -i\Omega_{i-1} + 2b\Omega_{i+1} + 2b(A-B)\Omega_i \\
&= 2p\Omega_{i+1} + 2b(A-B)\Omega_i - i\Omega_{i-1}
\end{align}
$$

Note that:

$$
2b(A-B) = -p(P-A)
$$

By the definition of $E^i_t$ it then follows that:

$$
\begin{equation}\label{eq:overlap-diff-p}
\begin{aligned}
\frac{\partial}{\partial P}\Omega_i &=
\sum_{t=0}^{i+1}2pE^{i+1}_t\Lambda_t - \sum_{t=0}^i 2p(P-A)E^i_t\Lambda_t -
\sum_{t=0}^{i-1}iE^{i-1}_t\Lambda_t \\
&= \sum_{t=0}^{i+1}\left(2pE^{i+1}_t - 2p(P-A)E^i_t - iE^{i-1}_t\right)\Lambda_t
\end{aligned}
\end{equation}
$$

Now we'll differentiate the right hand side of \ref{eq:cartesian-to-hermite} by $P$.

We claim that for all $i$ and $t$

$$
\frac{\partial}{\partial P}E^i_t(a, b, A, B) = 0
$$

To see this, we'll use the [overlap transform](#def:overlap-transform) to make a
change of coordinates from $(A,B)$ to $(P, Q)$ defined by:

$$
\begin{bmatrix}P \\ Q \end{bmatrix} = T_{ab} \begin{bmatrix}A \\ B \end{bmatrix}
= \begin{bmatrix}\frac{a}{p}A + \frac{b}{p}B \\ A - B \end{bmatrix}
$$

We'll now see that in the $(P, Q)$ coordinates, $E^i_t(a, b, P, Q)$ is a function of
$Q$ only. For the base case, note that:

$$
E^0_0 = K(a,b,A,B) = e^{-\frac{ab}{a + b} Q^2}
$$

Finally, note that the recurrence relation \ref{eq:cartesian-to-hermite-1}
depends only on:

$$
P - A = -\frac{b}{p}Q
$$

Since both $E^0_0$ and the recurrence \ref{eq:cartesian-to-hermite-1} depend only
on $Q$, so does $E^i_t$ for all $i$ and $t$.

We can now differentiate the right hand side of \ref{eq:cartesian-to-hermite} by $P$:

$$
\begin{equation}\label{eq:hermite-diff-p}
\begin{aligned}
\frac{\partial}{\partial P}\sum_{t=0}^iE^i_t\Lambda_t &=
\sum_{t=0}^iE^i_t\frac{\partial}{\partial P}\Lambda_t \\
&= \sum_{t=0}^iE^i_t\Lambda_{t+1} \\
&= \sum_{t=0}^{i+1}E^i_{t-1}\Lambda_t
\end{aligned}
\end{equation}
$$

Comparing the terms in equations \ref{eq:overlap-diff-p} and \ref{eq:hermite-diff-p}
we get:

$$
E^i_{t-1} = 2pE^{i+1}_t - 2p(P-A)E^i_t - iE^{i-1}_t
$$

which implies the second recurrence in the claim:

$$
\begin{equation}\label{eq:cartesian-to-hermite-2}
E^{i+1}_t = \frac{1}{2p}E^i_{t-1} + (P-A)E^i_t + \frac{i}{2p}E^{i-1}_t
\end{equation}
$$

The third recurrence in the claim follows by subtracting equations \ref{eq:cartesian-to-hermite-1}
and \ref{eq:cartesian-to-hermite-1}.

_q.e.d_
</div>
</details>


# Overlap Integrals



The simplest type of electron integral is the _overlap integral_:

> **Definition (Overlap Integral).** The overlap integral between the 
> Cartesian Gaussians $G_{\boldsymbol{\alpha}}(\mathbf{r}, a, \mathbf{A})$ and
> $G_{\boldsymbol{\beta}}(\mathbf{r}, a, \mathbf{A})$ is:
>
> $$
> \begin{align*}
> S_{\boldsymbol{\alpha}\boldsymbol{\beta}}(a,b,\mathbf{A},\mathbf{B}) :&=
> \langle G_{\boldsymbol{\alpha}}(\mathbf{r}, a, \mathbf{A}) | G_{\boldsymbol{\beta}}(\mathbf{r}, a, \mathbf{A}) \rangle \\
> &= \int_{\mathbb{R}^3} G_{\boldsymbol{\alpha}}(\mathbf{r}, a, \mathbf{A}) G_{\boldsymbol{\beta}}(\mathbf{r}, a, \mathbf{A}) d\mathbf{r}
> \end{align*}
> $$

Using our notation for overlaps:

$$
S_{\boldsymbol{\alpha}\boldsymbol{\beta}}(a,b,\mathbf{A},\mathbf{B}) = 
\int_{\mathbb{R}^3} \Omega_{\boldsymbol{\alpha}\boldsymbol{\beta}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B}) d\mathbf{r}
$$

The goal of this section is to prove the following set of recurrence relations which
determine $S_{\boldsymbol{\alpha}\boldsymbol{\beta}}$ for all multi-indices
$\boldsymbol{\alpha}$ and $\boldsymbol{\alpha}$.

> **Claim (Overlap Integral Base Case).**
> Let $a,b\in\mathbb{R}$ be positive real numbers, $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$.
> Then:
>
> $$
> S_{\mathbf{0}\mathbf{0}}(a,b,\mathbf{A},\mathbf{B}) 
> = \left(\frac{\pi}{p}\right)^{3/2} K(a,b,\mathbf{A},\mathbf{B}) 
> $$

> **Claim (Overlap Integral Recurrence).**
> Let $a,b\in\mathbb{R}$ be positive real numbers, $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$
> and $i,\alpha_y,\alpha_z,j,\beta_y,\beta_z\in\mathbb{Z}$ non-negative integers.
>
> Define:
>
> $$
> S_{ij} := S_{(i,\alpha_y,\alpha_z),(j,\beta_y,\beta_z)}(a, b, \mathbf{A},\mathbf{B})
> $$
>
> Then:
>
> 1. Vertical Transfer:
>
> $$
> S_{i+1, 0} = (P_x-A_x)S_{i,0} + \frac{i}{2p}S_{i-1, 0}
> $$
>
> 1. Horizontal Transfer:
>
> $$
> S_{i, j+1} = (A_x - B_x)S_{ij} + S_{i+1,j}
> $$
>
> And the analogous identities hold in the $y$ and $z$ coordinates.