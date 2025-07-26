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

We will frequently need to work with Cartesian coordinates relative to a center $\mathbf{A}\in\mathbb{R}^3$.

We'll denote a position $\mathbb{r}\in\mathbb{R}^3$ relative to $\mathbf{A}$ by:

$$
\mathbf{r}_A := \mathbf{r} - \mathbf{A}
$$

We'll denote the Cartesian coordinates of $\mathbf{r}_A$ by $x_A,y_A$ and $z_A$ and denote the
norm of $\mathbf{r}_A$ by $r_A$. For example,

$$
x_A := x - A_x
$$

and 

$$
r_A^2 = x_A^2 + y_A^2 + z_A^2
$$

> **Definition (Primitive Cartesian Gaussian).** Let $i,j,k\in\mathbb{N}$ be non-negative integers,
> $\mathbf{A}\in\mathbb{R}^3$ and $a\in\mathbb{R}\_>$ a positive real number.
> The primitive Cartesian Gaussian with _Cartesian quantum numbers_ $i,j,k$,
> _center_ $\mathbf{A}$ and _exponent_ $a$ is defined at position $\mathbf{r}=(x,y,z)\in\mathbb{R}^3$
> by:
>
> $$
> G_{i,j,k}(\mathbf{r}, a, \mathbf{A}) := (x - A_x)^i(y - A_y)^i (z - A_z)^k e^{-a ||\mathbf{r} - \mathbf{A}||^2}
> $$

An important property of Cartesian Gaussians is that they can be factorized along the Cartesian 
coordinates:

$$
G_{i,j,k}(\mathbf{r}, a, \mathbf{A}) = G_i(x, a, A_x)G_j(y, a, A_y)G_k(z, a, A_z)
$$

where for example:

$$
G_i(x, a, A_x) := (x - A_x)^i e^{-a(x - A_x)^2}
$$

## Gaussian Overlaps

> **Definition ()
One key reason that electron integrals of Cartesian Gaussians are easy to evaluate is that the
product of two Gaussians is another Gaussian:

> **Claim (Gaussian Product Rule).** Let $A_x,B_x\in\mathbb{R}$ be two centers and
> $a,b\in\mathbb{R}$ be two exponents. Then:
>
> $$
> e^{-a (x - A_x)^2}e^{-b (x - B_x)^2} = e^{-\mu X_{AB}}e^{p (x - P_x)^2}
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
> And the new center $P_x$ and _separation_ $X_{AB}$ are defined by:
>
> $$
> \begin{align*}
> P_x &= \frac{aA_x + bB_x}{p} \\
> X_{AB} &= A_x - B_x
> \end{align*}
> $$

Note that the factor $e^{-\mu X_{AB}}$ is constant with respect to the $x$
coordinate and will be denoted by:

$$
K_{ab}^x := e^{-\mu X_{AB}^2}
$$






