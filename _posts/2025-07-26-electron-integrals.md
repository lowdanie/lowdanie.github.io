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

{: #clm:overlap-vertical-transfer }
> **Claim (Overlap Vertical Transfer).** Let $\Omega_i$ be the one dimensional overlap
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

{: #clm:overlap-horizontal-transfer }
> **Claim (Overlap Horizontal Transfer).**
> Let $A,B\in\mathbb{R}$ be two positions,
> $a,b\in\mathbb{R}$ positive real numbers and $i,j\in\mathbb{Z}$ non-negative
> integers. Define:
>
> $$
> \Omega_{ij} := \Omega_{ij}(x, a, b, A, B)
> $$
>
> Then:
>
> $$
> \Omega_{i,j+1} = \Omega_{i+1,j} + (A - B)\Omega_{ij}
> $$

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

{: #clm:differentiation-product-bracket }
> **Claim (Differentiation Product Bracket).** For every positive
> $t\in\mathbb{Z}$:
>
> $$
> [\frac{\partial^t}{\partial x^t}, x] = t \frac{\partial^{t-1}}{\partial x^{t-1}}
> $$

{: #clm:hermite-gaussian-properties }
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

{: #defn:cartesian-overlap-to-hermite }
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

{: #clm:cartesian-overlap-to-hermite-recurrence }
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

Starting with the left side, by claim [Overlap Properties](#clm:overlap-vertical-transfer)
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

Therefore, by claim [Overlap Properties](#clm:overlap-vertical-transfer):

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
$\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$.

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
> and $\alpha_y,\alpha_z,\beta_y,\beta_z\in\mathbb{Z}$ non-negative integers.
> Let $p=a+b$ and $\mathbf{P}=\frac{a}{p}\mathbf{A} + \frac{b}{p}\mathbf{B}$.
>
> For non-negative integers $i,j\in\mathbb{Z}$ Define:
>
> $$
> S_{ij} := S_{(i,\alpha_y,\alpha_z),(j,\beta_y,\beta_z)}(a, b, \mathbf{A},\mathbf{B})
> $$
>
> Then:
>
> 1. Vertical Transfer:
>
>    $$
>    S_{i+1, 0} = (P_x-A_x)S_{i,0} + \frac{i}{2p}S_{i-1, 0}
>    $$
>
> 1. Horizontal Transfer:
>
>    $$
>    S_{i, j+1} = (A_x - B_x)S_{ij} + S_{i+1,j}
>    $$
>
> And the analogous identities hold in the $y$ and $z$ coordinates.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

First we'll prove the vertical transfer relation.

To facilitate notation, we'll define the Cartesian Overlaps:

$$
\begin{align*}
\Omega_i(x) &:= \Omega_{i,0}(x, a, b, A_x, B_x) \\
\Omega_{\alpha_y,\alpha_z}(y,z) &:=
\Omega_{\alpha_y,0}(y, a, b, A_y, B_y)\Omega_{\alpha_z,0}(z, a, b, A_z, B_z)
\end{align*}
$$

Hermite Gaussians:

$$
\begin{align*}
\Lambda^i_t(x) &:= \Lambda^i_t(x, p, P_x) \\
\Lambda_{uv}(y,z) &:= \Lambda^{\alpha_y}_t(y, p, P_y)\Lambda^{\alpha_z}_t(z, p, P_z)
\end{align*}
$$

and Cartesian to Hermite expansion coefficients:

$$
\begin{align*}
E^i_t &:= E^i_t(a, b, A_x, B_x) \\
E_{uv} &:= E^{\alpha_y}_u(a, b, A_y, B_y)E^{\alpha_z}_u(a, b, A_z, B_z)
\end{align*}
$$

By definitions [Cartesian Gaussian Overlap](#defn:cartesian-gaussian-overlap)
and [Cartesian Overlap To Hermite](#defn:cartesian-overlap-to-hermite):

$$
\begin{equation}\label{eq:overlap-integral-expansion}
\begin{aligned}
S_{i,0} &= 
\int_{\mathbb{R}^3} \Omega_{(i,\alpha_y,\alpha_z),\mathbf{0}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B}) d\mathbf{r} \\
&= \int_{\mathbb{R}^3} \Omega_i(x)\Omega_{\alpha_y,\alpha_z}(y,z) d\mathbf{r} \\
&= \sum_{t,u,v=0}^{i,\alpha_y,\alpha_z}E^i_t E_{uv}\int_{\mathbb{R}^3}\Lambda_t(x)\Lambda_{uv}(y,z)d\mathbf{r} \\
&=  \sum_{t,u,v=0}^{i,\alpha_y,\alpha_z}E^i_t E_{uv}
\left(\frac{\partial}{\partial P_x}\right)^t\left(\frac{\partial}{\partial P_y}\right)^u\left(\frac{\partial}{\partial P_z}\right)^v 
\int_{\mathbb{R}^3} G_\mathbf{0}(\mathbf{r}, p, P)d\mathbf{r}
\end{aligned}
\end{equation}
$$

By [Gaussian Integral](#clm:gaussian-integral):

$$
\int_{\mathbb{R}^3} G_\mathbf{0}(\mathbf{r}, p, P)d\mathbf{r} =  \left(\frac{\pi}{p}\right)^{3/2}
$$

Note that $\left(\frac{\pi}{p}\right)^{3/2}$ is independent of $\mathbf{P}$ which means
that all of the non-zero derivatives in equation \ref{eq:overlap-integral-expansion} vanish
and so:

$$
\begin{equation}\label{eq:overlap-integral-t-0}
S_{i,0} =  E^i_0 E_{0,0} \left(\frac{\pi}{p}\right)^{3/2}
\end{equation}
$$

We can now use
[Cartesian Overlap To Hermite Recurrence](#clm:cartesian-overlap-to-hermite-recurrence)
to derive a recurrence for $S_{i,0}$.

Specifically, note that by part 2 of that claim:

$$
E^{i+1}_0 = (P_x - A_x)E^i_0 + \frac{i}{2p}E^{i-1}_0
$$

Inserting this into equation \ref{eq:overlap-integral-t-0}
gives the vertical transfer relation.

The horizontal transfer relation follows immediately from
[Overlap Horizontal Transfer](#clm:overlap-horizontal-transfer).

_q.e.d_
</div>
</details>

# Kinetic Energy Integrals


# One Electron Coulomb Integrals

> **Definition (One Electron Coulomb Integral).**
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B},\mathbf{C}\in\mathbb{R}^3$
> and $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ multi-indexes.
>
> The one-electron integral between the Cartesian Gaussians
> $G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A})$ and
> $G_\boldsymbol{\beta}(\mathbf{r}, b, \mathbf{B})$ with center $\mathbf{C}$ is defined by:
>
> $$
> I_{\boldsymbol{\alpha},\boldsymbol{\beta}}(a, b, \mathbf{A}, \mathbf{B}, \mathbf{C}) :=
> \int_{\mathbb{R}^3}\frac{G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A})
> G_\boldsymbol{\beta}(\mathbf{r}, b, \mathbf{B})}{||\mathbf{r} - \mathbf{C}||}d\mathbf{r}
> $$

The goal of this section is to find recurrence relations for the one electron
integral $I_{\boldsymbol{\alpha},\boldsymbol{\beta}}(a, b, \mathbf{A}, \mathbf{B})$
as we vary the multi-indicies $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$.


First we'll assume that $\boldsymbol{\beta}=\mathbf{0}$ and focus on finding recuurence
relations for $\boldsymbol{\alpha}$.

As usual, set $p=a+b$ and $\mathbf{P} = \frac{a}{p}\mathbf{A} + \frac{b}{p}\mathbf{B}$.

By definitions [Cartesian Gaussian Overlap](#defn:cartesian-gaussian-overlap)
and [Cartesian Overlap To Hermite](#defn:cartesian-overlap-to-hermite):

$$
\begin{equation}\label{eq:one-electron-expansion}
\begin{aligned}
I_{\boldsymbol{\alpha},\mathbf{0}}(a, b, \mathbf{A}, \mathbf{B}, \mathbf{C}) &=
\int_{\mathbb{R}^3}\frac{\Omega_{\boldsymbol{\alpha},\mathbf{0}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B})}{||\mathbf{r}-\mathbf{C}||}d\mathbf{r} \\
&= \sum_{t,u,v=0}^{\boldsymbol{\alpha}}E^{\boldsymbol{\alpha}}_{tuv}(a, b, \mathbf{A}, \mathbf{B})
\int_{\mathbb{R}^3}\frac{\Lambda_{tuv}(\mathbf{r},p,\mathbf{P})}{||\mathbf{r}-\mathbf{C}||}d\mathbf{r} \\
&= \sum_{t,u,v=0}^{\boldsymbol{\alpha}}E^{\boldsymbol{\alpha}}_{tuv}(a, b, \mathbf{A}, \mathbf{B})
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
\int_{\mathbb{R}^3}\frac{G_\mathbf{0}(\mathbf{r},p,\mathbf{P})}{||\mathbf{r}-\mathbf{C}||}d\mathbf{r}
\end{aligned}
\end{equation}
$$

The point of this reformulation is that integrand no longer contains the 
multi-index $\boldsymbol{\alpha}$ and can be solved in close form using the
[Boys function](https://molssi.github.io/MIRP/boys_function.html).

{: #defn:boys-function }
> **Definition (Boys Function).**
> Let $n\in\mathbb{Z}$ be a non-negative integer 
> and $x\in\mathbb{R}$ a non-negative real number.
> The $n$-th Boys function is defined by:
>
> $$
> F_n(x) := \int_0^1 t^{2n}e^{-xt^2} dt
> $$

{: #clm:boys-derivative }
> **Claim (Boys Derivative).**
> Let $n\in\mathbb{Z}$ be a non-negative integer. Then
>
> $$
> \frac{d}{dx}F_n(x) = -F_{n+1}(x)
> $$

{: #clm:spherical-coulomb-integral }
> **Claim (Spherical Coulomb Integral).**
> Let $a\in\mathbb{R}$ be a positive integer and
> $\mathbf{A},\mathbf{C}\in\mathbb{R}^3$. Then:
>
> $$
> \int_{\mathbb{R}^3} \frac{G_\mathbf{0}(\mathbf{r}, a, \mathbf{A})}{||\mathbf{r} - \mathbf{C}||}d\mathbf{r}
> = \frac{2\pi}{a}F_0(a||\mathbf{A} - \mathbf{C}|| ^2)
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

We need to compute:

$$
\begin{equation}\label{eq:spherical-gaussian}
I := \int_{\mathbb{R}^3}\frac{e^{-a||\mathbf{r} - \mathbf{A}||^2}}{||\mathbf{r} - \mathbf{C}||}d\mathbf{r}
\end{equation}
$$

The main difficulty is that the denominator prevents us from factoring the integrand
into the three Cartesian coordinates as we could in the case of a regular
[Gaussian Integral](#clm:gaussian-integral).

We can work around this by expressing a fraction of the form $\frac{1}{x}$ in terms
of an integral of Gaussians.

Specifically, by the standard
[One Dimensional Gaussian Integral](#clm:one-dimensional-gaussian-integral),
for all positive $x\in\mathbb{R}$ we have:

$$
\int_{-\infty}^{\infty}e^{-x t^2}dt = \sqrt{\frac{\pi}{x}}
$$

Plugging in $x^2$ gives:

$$
\int_{-\infty}^{\infty}e^{-x^2t^2}dt = \sqrt{\frac{\pi}{x^2}} = \frac{\sqrt{\pi}}{x}
$$

Which implies:

$$
\begin{equation}\label{eq:reciprocal-to-exp}
\frac{1}{x} = \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty}e^{-x^2t^2}dt
\end{equation}
$$

Setting $x = ||\mathbf{r} - \mathbf{C}||$ and applying equation \ref{eq:reciprocal-to-exp} to
equation \ref{eq:spherical-gaussian}:

$$
I = \frac{1}{\sqrt{\pi}}\int_{\mathbb{R}^3}\int_{-\infty}^{\infty}
e^{-a||\mathbf{r} - \mathbf{A}||^2}
e^{-t^2||\mathbf{r}-\mathbf{C}||^2}
dt d\mathbf{r}
$$

By applying the [Gaussian Product Rule](#clm:gaussian-product-rule)
to $G(\mathbf{r}, a, \mathbf{A})$ and $G(\mathbf{r}, t^2, \mathbf{C})$:

$$
I = \frac{1}{\sqrt{\pi}}\int_{\mathbb{R}^3}\int_{-\infty}^{\infty}
e^{-\frac{at^2}{a + t^2}||\mathbf{A} - \mathbf{C}||^2}
e^{-(a + t^2)||\mathbf{r} - \mathbf{S}||^2}
dt d\mathbf{r}
$$

where:

$$
\mathbf{S} = \frac{1}{a + t^2}(a\mathbf{A} + t^2\mathbf{C})
$$

Integrating over $\mathbf{r}$ and applying [Gaussian Integral](#clm:gaussian-integral):

$$
\begin{align*}
I &= \pi \int_{-\infty}^{\infty}
(a + t^2)^{-3/2} e^{-\frac{at^2}{a + t^2}||\mathbf{A} - \mathbf{C}||^2} dt \\
&= 2\pi \int_{0}^{\infty}
(a + t^2)^{-3/2} e^{-\frac{at^2}{a + t^2}||\mathbf{A} - \mathbf{C}||^2} dt
\end{align*}
$$

We now use the variable substitution:

$$
\begin{align*}
u^2 &= \frac{t^2}{a + t^2} \\
du &= a(a + t^2)^{-3/2}dt
\end{align*}
$$

and get:

$$
\begin{align*}
I &= \frac{2\pi}{a}\int_0^1 e^{-a||\mathbf{A} - \mathbf{C}||^2u^2}du \\
&= \frac{2\pi}{a} F_0(a||\mathbf{A} - \mathbf{C}||^2)
\end{align*}
$$

_q.e.d_

</div>
</details>

We'll collect the partial derivatives of the Boys function in equation \ref{eq:one-electron-expansion}
into a function that we can analyze independently:

{: #defn:boys-partial-derivatives }
> **Definition (Boys Partial Derivatives).**
> Let $s\in\mathbb{R}$ be a positive real number,
> $\mathbf{C},\mathbf{P}\in\mathbb{R}^3$.
> For all non-negative integers $t,u,v,n\in\mathbb{Z}$ define:
>
> $$
> R_{tuv}^n(s, \mathbf{C}, \mathbf{P}) := (-2s)^n 
> \left(\frac{\partial}{\partial P_x}\right)^t
> \left(\frac{\partial}{\partial P_y}\right)^u
> \left(\frac{\partial}{\partial P_z}\right)^v
> F_n(s||\mathbf{P} - \mathbf{C}||^2)
> $$

We'll also define a generalization of equation \ref{eq:one-electron-expansion}
that will simplify the form of the recurrence relations and also be applicable to
the 2-electron case:

{: #defn:nth-order-coulomb-integral }
> **Definition ($n$-th Order Coulomb Integral).**
> Let $a,b,s\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B},\mathbf{C}\in\mathbb{R}^3$.
>
> Let $p=a+b$ and $\mathbf{P} = \frac{a}{p}\mathbf{A} + \frac{b}{p}\mathbf{B}$.
>
> For all non-negative integers $i,j,k,n\in\mathbb{Z}$ define:
>
> $$
> V_{ijk}^n(a,b,s,\mathbf{A},\mathbf{B},\mathbf{C}) :=
> (-2s)^{-n}
> \sum_{t,u,v=0}^{ijk}E^{ijk}_{tuv}(a, b, \mathbf{A}, \mathbf{B})
> R_{tuv}(s, \mathbf{C}, \mathbf{P})
> $$

Note that, recalling that $p=a+b$, we can rewrite equation \ref{eq:one-electron-expansion} as

$$
\begin{equation}\label{eq:one-electron-expansion-2}
I_{(i,j,k),\mathbf{0}}(a, b, \mathbf{A}, \mathbf{B}, \mathbf{C}) =
\frac{2\pi}{p} V_{ijk}^0(a, b, p, \mathbf{A}, \mathbf{B}, \mathbf{C})
\end{equation}
$$

We'll now find recurrence relations for the $n$-th order Coulomb Integrals $V_{ijk}^n$
in two steps. First we'll use [Boys Derivative](#clm:boys-derivative) to find recurrence
relations for the [Boys Partial Derivatives](#defn:boys-partial-derivatives) $R_{tuv}^n$.
Next we'll use the
[Cartesian Overlap To Hermite Recurrence](#clm:cartesian-overlap-to-hermite-recurrence)
on the $E_{tuv}^{ijk}$ coefficients to derive recurrence relations for  $V_{ijk}^n$.

{: #clm:boys-partial-derivative-recurrence }
> **Claim (Boys Partial Derivative Recurrence).**
> Let $s\in\mathbb{R}$ be a positive real number,
> $\mathbf{C},\mathbf{P}\in\mathbb{R}^3$.
> For all non-negative integers $t,u,v,n\in\mathbb{Z}$ define:
>
> $$
> R_{tuv}^n := R^n_{tuv}(s, \mathbf{C}, \mathbf{P})
> $$
>
> Then:
>
> 1. Base Case:
> 
>    $$
>    R_{000}^n = (-2s)^nF_n(s||\mathbf{P} - \mathbf{C}||)
>    $$
>
> 2. Vertical Transfer ($x$ coordinate):
>
>    $$
>    R^n_{t+1,u,v} = (P_x - C_x)R^{n+1}_{tuv} + t R^{n+1}_{t-1,u,v}
>    $$
>
> And the analogous vertical transfer relation holds for the $y$ and $z$
> coordinates.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The base case follows immediately from the definition.

The proof of the vertical transfer relation follows by direct calculation. 
First, by definition:

$$
\begin{align*}
R^n_{t+1,u,v} &= (-2s)^n
\left(\frac{\partial}{\partial P_x}\right)^{t+1}
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
F_n(s||\mathbf{P} - \mathbf{C}||^2) \\
&= (-2s)^n
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
\frac{\partial}{\partial P_x}
F_n(s||\mathbf{P} - \mathbf{C}||^2)
\end{align*}
$$

Applying [Boys Derivative](#clm:boys-derivative) gives:

$$
R^n_{t+1,u,v} = (-2s)^n
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
(-2s)(P_x - C_x)
F_n(s||\mathbf{P} - \mathbf{C}||^2) 
$$

We now use [Differentiation Product Bracket](#clm:differentiation-product-bracket)
to move the $(-2s)(P_x - C_x)$ term to the front:

$$
\begin{align*}
R^n_{t+1,u,v} &= (-2s)^{n+1}(P_x - C_x)
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
F_n(s||\mathbf{P} - \mathbf{C}||^2) \\
&+ t(-2s)^{n+1}
\left(\frac{\partial}{\partial P_x}\right)^{t-1}
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
F_n(s||\mathbf{P} - \mathbf{C}||^2)  \\
&= (P_x - C_x)R^{n+1}_{tuv} + t R^{n+1}_{t-1,u,v}
\end{align*}
$$

_q.e.d_
</div>
</details>

We can now combine this result with the
[Cartesian Overlap To Hermite Recurrence](#clm:cartesian-overlap-to-hermite-recurrence)
 to derive recurrence relations for  $V_{ijk}^n$.

{: #clm:nth-order-coulomb-recurrence }
> **Claim ($n$-th Order Coulomb Recurrence)**
> Let $a,b,s\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B},\mathbf{C}\in\mathbb{R}^3$.
>
> Let $p=a+b$ and $\mathbf{P} = \frac{a}{p}\mathbf{A} + \frac{b}{p}\mathbf{B}$.
>
> For all non-negative integers $i,j,k,n\in\mathbb{Z}$ define:
>
> $$
> V_{ijk}^n := V_{ijk}^n(a,b,s,\mathbf{A},\mathbf{B},\mathbf{C})
> $$
>
> Then:
>
> 1. Base case:
>    
>    $$
>    V_{000}^n = K(a,b,\mathbf{A},\mathbf{B})F_n(p||\mathbf{P} - \mathbf{C}||)
>    $$
>
> 2. Vertical Transfer ($x$ coordinate):
>
>    $$
>    V_{i+1,j,k}^n = (P_x - A_x)V_{ijk}^n - \frac{s}{p}(P_x - C_x)V_{ijk}^{n+1} 
>    + \frac{i}{2p}(V_{i-1,j,k}^n - \frac{s}{p}V_{i-1,j,k}^{n+1})
>    $$
>
> And the analogous vertical transfer relation holds for the $y$ and $z$
> coordinates.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The base case follows from the definition and the base case of 
[Cartesian Overlap To Hermite Recurrence](#clm:cartesian-overlap-to-hermite-recurrence).

We'll now prove the vertical transfer relation. To facilitate notation, we'll
define:

$$
\begin{align*}
E_t^i &:= E_t^i(a, b, A_x, B_x) \\
E_{uv}^{jk} &:= E_u^j(a, b, A_y, B_y)E_v^k(a, b, A_z, B_z)
\end{align*}
$$

and:

$$
R_{tuv}^n := R_{tuv}^n(p, \mathbf{C}, \mathbf{P})
$$

By the definition of $V_{ijk}^n$:

$$
\begin{equation}\label{eq:nth-order-coulomb-rr-1}
V_{i+1,jk}^n = (-2s)^{-n}\sum_{tuv}^{i+1,jk}E_t^{i+1}E_{uv}^{jk}R_{tuv}^n
\end{equation}
$$

By
[Cartesian Overlap To Hermite Recurrence](#clm:cartesian-overlap-to-hermite-recurrence):

$$
E^{i+1}_t = \frac{1}{2p}E^i_{t-1} + (P_x-A_x)E^i_t + \frac{i}{2p}E^{i-1}_t
$$

Inserting this into equation \ref{eq:nth-order-coulomb-rr-1}:

$$
\begin{align*}
V_{i+1,jk}^n &= 
(-2s)^{-n}\sum_{tuv}^{i+1,jk}\frac{1}{2p}E_{t-1}^iE_{uv}^{jk}R_{tuv}^n \\
&+
(-2s)^{-n}\sum_{tuv}^{i+1,jk}(P_x-A_x)E_t^iE_{uv}^{jk}R_{tuv}^n \\
&+
(-2s)^{-n}\sum_{tuv}^{i+1,jk}\frac{i}{2p}E_t^{i-1}E_{uv}^{jk}R_{tuv}^n
\end{align*}
$$

Using the fact that $E_t^i = 0$ when $t<0$ or $t>i$ we can rewrite this as:

$$
\begin{equation}\label{eq:nth-order-coulomb-rr-2}
\begin{aligned}
V_{i+1,jk}^n &= 
(-2s)^{-n}\sum_{tuv}^{ijk}\frac{1}{2p}E_t^iE_{uv}^{jk}R_{t+1,uv}^n \\
&+
(-2s)^{-n}\sum_{tuv}^{ijk}(P_x-A_x)E_t^iE_{uv}^{jk}R_{tuv}^n \\
&+
(-2s)^{-n}\sum_{tuv}^{i-1,jk}\frac{i}{2p}E_t^{i-1}E_{uv}^{jk}R_{tuv}^n
\end{aligned}
\end{equation}
$$

The second two terms on the right hand side of equation \ref{eq:nth-order-coulomb-rr-2}
are equal to $(P_x - A_x)V_{ijk}^n$ and $\frac{i}{2p}V_{i-1,jk}^n$ respectively.

To handle the first term, we'll use the
[Boys Partial Derivative Recurrence](clm:boys-partial-derivative-recurrence):

$$
R_{t+1,uv}^n = (P_x - C_x)R_{tuv}^{n+1} + t R_{t-1,u,v}^{n+1}
$$

Inserting this into the first term of equation \ref{eq:nth-order-coulomb-rr-2}:

$$
\begin{align*}
\sum_{tuv}^{ijk}\frac{1}{2p}E_t^iE_{uv}^{jk}R_{t+1,uv}^n &=
\sum_{tuv}^{ijk}\frac{1}{2p}(P_x - C_x)E_t^iE_{uv}^{jk}R_{tuv}^{n+1} \\
&+ \sum_{tuv}^{ijk}\frac{t}{2p}E_t^iE_{uv}^{jk}R_{t-1,uv}^{n+1}
\end{align*}
$$

Re-indexing $t$ in the second term:

$$
\begin{align*}
\sum_{tuv}^{ijk}\frac{1}{2p}E_t^iE_{uv}^{jk}R_{t+1,uv}^n
&= \sum_{tuv}^{ijk}\frac{1}{2p}(P_x - C_x)E_t^iE_{uv}^{jk}R_{tuv}^{n+1} \\
&+ \sum_{tuv}^{i-1,jk}\frac{t+1}{2p}E_{t+1}^iE_{uv}^{jk}R_{t,uv}^{n+1}
\end{align*}
$$

According to part 3 of
[Cartesian Overlap To Hermite Recurrence](#clm:cartesian-overlap-to-hermite-recurrence):

$$
(t+1)E_{t+1}^i = \frac{i}{2p}E_t^{i-1}
$$

Therefore:

$$
\begin{align*}
\sum_{tuv}^{ijk}\frac{1}{2p}E_t^iE_{uv}^{jk}R_{t+1,uv}^n
&= \sum_{tuv}^{ijk}\frac{1}{2p}(P_x - C_x)E_t^iE_{uv}^{jk}R_{tuv}^{n+1} \\
&+ \sum_{tuv}^{i-1,jk}\frac{i}{(2p)^2}E_t^{i-1}E_{uv}^{jk}R_{tuv}^{n+1}
\end{align*}
$$

Inserting this back into equation \ref{eq:nth-order-coulomb-rr-2}:

$$
\begin{align*}
V_{i+1,jk}^n &= 
-\frac{s}{p}(P_x-C_x)(-2s)^{-{n+1}}\sum_{tuv}^{ijk}E_t^iE_{uv}^{jk}R_{tuv}^{n+1} \\
&- 
\frac{is}{2p^2}(-2s)^{-{n+1}}\sum_{tuv}^{i-1,jk}E_t^{i-1}E_{uv}^{jk}R_{tuv}^{n+1} \\
&+
(P_x-A_x)(-2s)^{-n}\sum_{tuv}^{ijk}E_t^iE_{uv}^{jk}R_{tuv}^n \\
&+
\frac{i}{2p}(-2s)^{-n}\sum_{tuv}^{i-1,jk}E_t^{i-1}E_{uv}^{jk}R_{tuv}^n
\end{align*}
$$

By the definition of $V_{ijk}^n$ we can rewrite this as:

$$
V_{i+1,jk}^n = -\frac{s}{p}(P_x - C_x)V_{ijk}^{n+1} - \frac{is}{2p^2}V_{i-1,jk}^{n+1} +
(P_x - A_x)V_{ijk}^n + \frac{i}{2p}V_{i-1,jk}^n
$$

Rearranging terms gives the desired result:

$$
V_{i+1,j,k}^n = (P_x - A_x)V_{ijk}^n - \frac{s}{p}(P_x - C_x)V_{ijk}^{n+1} +
\frac{i}{2p}(V_{i-1,j,k}^n - \frac{s}{p}V_{i-1,j,k}^{n+1})
$$

_q.e.d_

</div>
</details>

The [$n$-th Order Coulomb Recurrence](#clm:nth-order-coulomb-recurrence) can be used to
compute $V_{ijk}^n(a,b,s,\mathbf{A},\mathbf{B},\mathbf{C})$ for all
non-negative $i,j,k,n\in\mathbb{Z}$. We'll conclude this section by showing
how to use the $n$-th order Coulomb integral to compute the
[One Electron Coulomb Integral](#defn:one-electron-coulomb-integral)

> **Claim (One Electron Coulomb Base Case).**
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B},\mathbf{C}\in\mathbb{R}^3$ and
> $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ multi-indices. Then:
>
> $$
> I_{\boldsymbol{\alpha},\boldsymbol{\beta}}(a, b, \mathbf{A},\mathbf{B},\mathbf{C}) =
> \frac{2\pi}{a+b}V_\boldsymbol{\alpha}^0(a, b, a+b, \mathbf{A},\mathbf{B},\mathbf{C})
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The claim follows directly from equation \ref{eq:one-electron-expansion-2}.

_q.e.d_
</div>
</details>

> **Claim (One Electron Coulomb Horizontal Transfer).**
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B},\mathbf{C}\in\mathbb{R}^3$.
>
> For all multi-indices
> $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ define:
>
> $$
> I_{\boldsymbol{\alpha},\boldsymbol{\beta}} := 
> I_{\boldsymbol{\alpha},\boldsymbol{\beta}}(a, b, \mathbf{A},\mathbf{B},\mathbf{C})
> $$
>
> Then:
>
> $$
> I_{\boldsymbol{\alpha},(\beta_x+1,\beta_y,\beta_z)} = 
> (A_x - B_x)I_{\boldsymbol{\alpha},\boldsymbol{\beta}} +
> I_{(\alpha_x+1,\alpha_y,\alpha_z),\boldsymbol{\beta}}
> $$
>
> And the analogous relation holds for the $y$ and $z$ coordinates.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The claim follows from the definition of the
[One Electron Coulomb Integral](#defn:one-electron-coulomb-integral) and
[Overlap Horizontal Transfer](#clm:overlap-horizontal-transfer).

_q.e.d_
</div>
</details>

# Two Electron Coulomb Integrals

> **Definition (Two Electron Coulomb Integral).**
> Let $a,b,c,d\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}\in\mathbb{R}^3$
> and 
> $\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}\in\mathbb{Z}^3$
> multi-indices.
>
> The two-electron integral is defined as:
>
> $$
> J_{\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}}
> (a, b, c, d, \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}) :=
> \int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
> \frac{
> G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A})
> G_\boldsymbol{\beta}(\mathbf{r}, b, \mathbf{B})
> G_\boldsymbol{\gamma}(\mathbf{r}, c, \mathbf{C})
> G_\boldsymbol{\delta}(\mathbf{r}, d, \mathbf{D})
> }{||\mathbf{r}_1 - \mathbf{r}_2||}
> d\mathbf{r}_1d\mathbf{r}_2
> $$

Despite looking more formidable than the one-electron integral, 
when $\boldsymbol{\beta}=\boldsymbol{\gamma}=\boldsymbol{\delta}=\mathbf{0}$, 
the two-electron integral can also be expressed in terms of the
[$n$-th order Coulomb integral](defn:nth-order-coulomb-integral).
We can then use an analog of the one-electron horizontal transfer recurrence
and a new _electron transfer_ recurrence to extend to the general case.

We'll start by analyzing the case where
$\boldsymbol{\beta}=\boldsymbol{\gamma}=\boldsymbol{\delta}=\mathbf{0}$.
A key result relating the two-electron integral to the Boys function is:

{: #clm:spherical-two-electron-coulomb-integral }
> **Claim (Spherical Two Electron Coulomb Integral).**
> Let $p,q\in\mathbb{R}$ be positive real numbers and
> $\mathbf{P},\mathbf{Q}\in\mathbb{R}^3$. 
>
> Let $s = \frac{pq}{p+q}$. Then:
>
> $$
> \int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
> \frac{
> G_\mathbf{0}(\mathbf{r}_1, p, \mathbf{P})
> G_\mathbf{0}(\mathbf{r}_2, q, \mathbf{Q})
> }{||\mathbf{r}_1 - \mathbf{r}_2||}
> d\mathbf{r}_1d\mathbf{r}_2 = 
> \frac{2\pi^{5/2}}{pq\sqrt{p+q}}F_0(s ||\mathbf{P}-\mathbf{Q}||^2)
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

Let's first define the following spherical Coulomb integral:

$$
S(\mathbf{r}_2) := 
\int_{\mathbf{R}^3}
\frac{
G_\mathbf{0}(\mathbf{r}_1, p, \mathbf{P})
}{||\mathbf{r}_1 - \mathbf{r}_2||}
d\mathbf{r}_1
$$

We can rewrite the spherical two-electron Coulomb integral as:

$$
\int_{\mathbb{R}^3}
S(\mathbf{r}_2)
G_\mathbf{0}(\mathbf{r}_2, q, \mathbf{Q})
d\mathbf{r}_2
$$

By [Spherical Coulomb Integral](#clm:spherical-coulomb-integral)
this is equal to:

$$
\frac{2\pi}{p}\int_{\mathbb{R}^3}
F_0(p||\mathbf{P} - \mathbf{r}_2||^2)
G_\mathbf{0}(\mathbf{r}_2, q, \mathbf{Q})
d\mathbf{r}_2
$$

Substituting the definition of the [Boys Function](#defn:boys-function) gives:

$$
\frac{2\pi}{p}
\int_{\mathbb{R}^3}
\int_0^1 
G_\mathbf{0}(\mathbf{r}_2, pt^2, \mathbf{P})
G_\mathbf{0}(\mathbf{r}_2, q, \mathbf{Q})
dt
d\mathbf{r}_2
$$

We now apply the [Gaussian Product Rule](#clm:gaussian-product-rule) to obtain:

$$
\frac{2\pi}{p}
\int_{\mathbb{R}^3}
\int_0^1
K(pt^2, q, \mathbf{P}, \mathbf{Q})
G_\mathbf{0}(\mathbf{r}_2, s(t), \mathbf{S}(t))
dt
d\mathbf{r}_2
$$

where

$$
\begin{align*}
s(t) &= pt^2 + q \\
\mathbf{S}(t) &= \frac{pt^2}{s}\mathbf{P} + \frac{q}{s}\mathbf{Q} \\
K(pt^2, q, \mathbf{P}, \mathbf{Q}) &= e^{-\frac{pqt^2}{pt^2 + q}||\mathbf{P}-\mathbf{Q}||^2}
\end{align*}
$$

We now evaluate the integral over $\mathbf{r}_2$ using
[Gaussian Integral](#clm:gaussian-integral):

$$
\frac{2\pi}{p} \pi^{3/2}
\int_0^1 
\frac{1}{(pt^2 + q)^{3/2}}
e^{-\frac{pqt^2}{pt^2 + q}||\mathbf{P}-\mathbf{Q}||^2}
dt
$$

To evaluate the integral over $t$ we use the substitution:

$$
u^2 = \frac{(p+q)t^2}{pt^2 + q}
$$

Taking derivatives of both sides together with a bit of algebra gives:

$$
\frac{1}{q\sqrt{p+q}}du = \frac{1}{(pt^2 + q)^{3/2}}dt
$$

Therefore, after substituting $u$ we have:

$$
\frac{2\pi^{5/2}}{pq\sqrt{p+q}}
\int_0^1 
e^{-\frac{pq}{p+q} u^2||\mathbf{P}-\mathbf{Q}||^2}
du
$$

which, by the definition of the [Boys Function](#defn:boys-function), is equal to:

$$
\frac{2\pi^{5/2}}{pq\sqrt{p+q}}F_0(\frac{pq}{p+q}||\mathbf{P}-\mathbf{Q}||^2)
$$

_q.e.d_

</div>
</details>

We can now prove the reduction from the two-electron integral to the
$n$-th order Coulomb Integral:

{: #clm:two-electron-coulomb-base-case}
> **Claim (Two Electron Coulomb Base Case).**
> Let $a,b,c,d\in\mathbb{R}$ be positive real numbers and
> $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}\in\mathbb{R}^3$.
>
> Let $p=a+b$, $q=c+d$, $s = \frac{pq}{p+q}$ and:
>
> $$
> \begin{align*}
> \mathbf{P} &=\frac{a}{p}\mathbf{A}+\frac{b}{p}\mathbf{B} \\
> \mathbf{Q} &=\frac{c}{q}\mathbf{C}+\frac{d}{q}\mathbf{D}
> \end{align*}
> $$
>
> Then:
>
> $$
> J_{\boldsymbol{\alpha},\mathbf{0},\mathbf{0},\mathbf{0}}
> (a, b, c, d, \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}) =
> \frac{2\pi^{5/2}}{pq\sqrt{p+q}}K(c, d, \mathbf{C}, \mathbf{D})
> V_\boldsymbol{\alpha}^0(a, b, s, \mathbf{A}, \mathbf{B}, \mathbf{Q})
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

To facilitate notation, we'll define:

$$
J_{\boldsymbol{\alpha}} := 
J_{\boldsymbol{\alpha},\mathbf{0},\mathbf{0},\mathbf{0}}
(a, b, c, d, \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D})
$$

By the [Gaussian Product Rule](#clm:gaussian-product-rule):

$$
G_\mathbf{0}(\mathbf{r}, c, \mathbf{C})
G_\mathbf{0}(\mathbf{r}, d, \mathbf{D}) =
K(c, d, \mathbf{C}, \mathbf{D})G_\mathbf{0}(\mathbf{r}, q, \mathbf{Q})
$$

By the definition of the
[Cartesian Gaussian Overlap](#defn:cartesian-gaussian-overlap)
we can rewrite the two-electron integral as:

$$
J_{\boldsymbol{\alpha}} = 
K(c, d, \mathbf{C}, \mathbf{D})
\int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
\frac{
\Omega_{\boldsymbol{\alpha}, \mathbf{0}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B})
G_\mathrm{0}(\mathbf{r}, q, \mathbf{Q})
}{||\mathbf{r}_1 - \mathbf{r}_2||}
d\mathbf{r}_1d\mathbf{r}_2
$$

We'll now use 
[Cartesian Overlap To Hermite](#defn:cartesian-overlap-to-hermite)
to expand 
$\Omega_{\boldsymbol{\alpha}, \mathbf{0}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B})$
in terms of Hermite Gaussians:

$$
J_{\boldsymbol{\alpha}} = 
K(c, d, \mathbf{C}, \mathbf{D})
\sum_{tuv}^\boldsymbol{\alpha}
E_{tuv}^\boldsymbol{\alpha}(a, b, \mathbf{A}, \mathbf{B})
\int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
\frac{
\Lambda_{tuv}(\mathbf{r}, p, \mathbf{P})
G_\mathrm{0}(\mathbf{r}, q, \mathbf{Q})
}{||\mathbf{r}_1 - \mathbf{r}_2||}
d\mathbf{r}_1d\mathbf{r}_2
$$

By the definition of
[Hermite Gaussians](#defn:hermite-gaussian):

$$
J_{\boldsymbol{\alpha}} = 
K(c, d, \mathbf{C}, \mathbf{D})
\sum_{tuv}^\boldsymbol{\alpha}
E_{tuv}^\boldsymbol{\alpha}(a, b, \mathbf{A}, \mathbf{B})
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
\int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
\frac{
G_\mathbf{0}(\mathbf{r}, p, \mathbf{P})
G_\mathbf{0}(\mathbf{r}, q, \mathbf{Q})
}{||\mathbf{r}_1 - \mathbf{r}_2||}
d\mathbf{r}_1d\mathbf{r}_2
$$

Applying the 
[Spherical Two Electron Coulomb Integral](#clm:spherical-two-electron-coulomb-integral):

$$
J_{\boldsymbol{\alpha}} = 
\frac{2\pi^{5/2}}{pq\sqrt{p+q}}
K(c, d, \mathbf{C}, \mathbf{D})
\sum_{tuv}^\boldsymbol{\alpha}
E_{tuv}^\boldsymbol{\alpha}(a, b, \mathbf{A}, \mathbf{B})
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
F_0(s||\mathbf{P} - \mathbf{Q}||^2)
$$

The claim now follows from the definition of the
[$n$-th order Coulomb intergal](#defn:nth-order-coulomb-integral).

_q.e.d_
</div>
</details>

{: #clm:two-electron-coulomb-horizontal-transfer }
> **Claim (Two Electron Coulomb Horizontal Transfer).**
> Let $a,b,c,d\in\mathbb{R}$ be positive real numbers and
> $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}\in\mathbb{R}^3$.
>
> For all multi-indices
> $\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}\in\mathbb{Z}^3$
> define:
>
> $$
> J_{\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}} :=
> J_{\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}}
> (a, b, c, d, \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D})
> $$
>
> Then, the following relations hold:
>
> 1. $$
>    J_{\boldsymbol{\alpha},(\beta_x+1,\beta_y\beta_z),\boldsymbol{\gamma},\boldsymbol{\delta}} =
>    (A_x - B_x) J_{\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}} +
>    J_{(\alpha_x+1,\alpha_y\alpha_z),\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}}
>    $$
>
> 2. $$
>    J_{\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},(\delta_x+1,\delta_y\delta_z)} =
>    (C_x - D_x) J_{\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\gamma},\boldsymbol{\delta}} +
>    J_{\boldsymbol{\alpha},\boldsymbol{\beta},(\gamma_x+1,\gamma_y\gamma_z),\boldsymbol{\delta}}
>    $$
>
> And the analogous relations hold in the $y$ and $z$ coordinates.

{: #clm:two-electron-coulomb-electron-transfer }
> **Claim (Two Electron Coulomb Electron Transfer).**
> Let $a,b,c,d\in\mathbb{R}$ be positive real numbers and
> $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}\in\mathbb{R}^3$.
>
> Let $p=a+b$ and $q=c+d$.
>
> For all non-negative integers 
> $i,j,\alpha_y,\alpha_z,\gamma_y,\gamma_z\in\mathbb{Z}$
> define:
>
> $$
> J_{ij} :=
> J_{(i,\alpha_y,\alpha_z),\mathbf{0},(j,\gamma_y,\gamma_z),\mathbf{0}}
> (a, b, c, d, \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D})
> $$
>
> Then:
>
> $$
> J_{i,j+1} =
> -\frac{1}{q}(b(A_x - B_x) + d(C_x - D_x))J_{i,j} +
> \frac{i}{2q}J_{i-1,j} +\frac{j}{2q}J_{i,j-1} - \frac{p}{q}J_{i+1,j}
> $$
>
> And the analogous relations hold in the $y$ and $z$ coordinates.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

Consider the quantity $J_{ij}$ as a function of the $x$-coordinates
$A_x,B_x,C_x,D_x\in\mathbb{R}$ where the $y$ and $z$ coordinates of
$\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}\in\mathbb{R}^3$ are held fixed:

$$
J_{ij}(A_x,B_x,C_x,D_x) :=
J_{(i,\alpha_y,\alpha_z),\mathbf{0},(j,\gamma_y,\gamma_z),\mathbf{0}}
(a, b, c, d, (A_x,A_y,A_z), (B_x,B_y,B_z), (C_x,C_y,C_z), (D_x,D_y,D_z))
$$

Since the
[Two Electron Coulomb Integral](#defn:two-electron-coulomb-integral) is defined as an
integral over all of space,
$J_{ij}(A_x,B_x,C_x,D_x)$
is invariant to translations of the $x$-axis $\mathbb{R}$. More precisely, for any scalar
$v\in\mathbb{R}^3$:

$$
J_{ij}(A_x+v,B_x+v,C_x+v,D_x+v) =
J_{ij}(A_x,B_x,C_x,D_x)
$$

This implies that, for all $v\in\mathbb{R}$,
the directional derivative of $J_{ij}(A_x,B_x,C_x,D_x)$ in the direction 
$\mathbf{e} := (1, 1, 1, 1) \in \mathbb{R}^4$ is zero:

$$
\nabla_\mathbf{e}J_{ij}(A_x,B_x,C_x,D_x) = 0
$$

By the [standard](https://en.wikipedia.org/wiki/Directional_derivative#For_differentiable_functions)
relationship between directional derivatives and the gradient, this implies that:

$$
\mathbf{e} \dot \nabla J_{ij}(A_x,B_x,C_x,D_x) = 0
$$

By the definition of the gradient this implies:

$$
(\frac{\partial}{\partial A_x} + 
 \frac{\partial}{\partial B_x} + 
 \frac{\partial}{\partial C_x} + 
 \frac{\partial}{\partial D_x})
 J_{ij}(A_x,B_x,C_x,D_x) = 0
$$

By using the definition of the [Two Electron Coulomb Integral](#defn:two-electron-coulomb-integral)
and [Two Electron Coulomb Electron Transfer](#clm:two-electron-coulomb-electron-transfer):

$$
\begin{align*}
\frac{\partial}{\partial A_x}J_{ij} &= 2a J_{i+1,j} - iJ_{i-1,j} \\
\frac{\partial}{\partial B_x}J_{ij} &= 2b(A_x - B_x)J_{i,j} + 2bJ_{i+1,j} \\
\frac{\partial}{\partial C_x}J_{ij} &= 2c J_{i,j+1} - jJ_{i,j-1}\\
\frac{\partial}{\partial D_x}J_{ij} &= 2d(C_x - D_x)J_{i,j} + 2dJ_{i,j+1} \\
\end{align*}
$$

Equating the sum of these terms to zero and rearranging the gives the desired result.

_q.e.d_

</div>
</details>
