---
layout: note
title: "Density Functional Theory"
mathjax: true
---

In this note we follow the definitions from the [Hartree Fock](/blog/2025/07/26/scf)
post.

# The Density Function

Recall that the single electron state space is defined as

$$
\mathcal{H} := L^2(\mathbb{R}^3)\otimes\mathbb{C}^2
$$

and that the $n$-electron state space is defined as the exterior product

$$
\Lambda^n\mathcal{H}
$$

The _single-electron local density operator_
$N^1(\mathbf{r})\in\mathrm{End}(\mathcal{H})$ acts as a spatial projector (i.e a Dirac delta)
and acts trivially on the spin component:

$$
N^1(\mathbf{r}) = |\mathbf{r}\rangle\langle\mathbf{r}|\otimes\mathrm{Id}
$$

Specifically $N^1(\mathbf{r})$ acts on a decomposable state
$\|\psi\rangle\otimes\|s\rangle$ by:

$$
N^1(\mathbf{r}) \cdot |\psi\rangle\otimes|s\rangle := 
\psi(\mathbf{r})|\mathbf{r}\rangle\otimes|s\rangle
$$

To see why this is called the local density operator, note that the expectation
value of $N^1(\mathbf{r})$ for a state $\chi = \|\psi\rangle\otimes\|s\rangle$
is equal to:

$$
\begin{align*}
\langle \chi | N^1(\mathbf{r}) | \chi \rangle
&=\langle \psi|\mathbf{r}\rangle\langle\mathbf{r}|\psi\rangle \langle s | s \rangle \\
&= \psi(\mathbf{r})^*\psi(\mathbf{r}) \\
&= |\psi(\mathbf{r})|^2
\end{align*}
$$

In other words, the expectation value of $N^1(\mathbf{r})$ is equal to the
probability density of finding the electron at $\mathbf{r}\in\mathbb{R}^3$.

In general, we define the _density function_ $\rho\in\mathrm{Hom}(\mathbb{R}^3,\mathbb{R})$
associated to a state $\chi\in\mathcal{H}$ in terms of the expectation of the local density operator:

$$
\rho(\mathbf{r}) := \langle \chi | N^1(\mathbf{r}) | \chi \rangle
$$

The _$n$-electron local density operator_ $N^n(\mathbf{r})\in\mathrm{End}(\Lambda^n\mathcal{H})$
is defined to be the symmetric extension (or derivation) of $N^1(\mathbf{r})$ from 
$\mathrm{End}(\mathcal{H})$ to $\mathrm{End}(\Lambda^n\mathcal{H})$.

Similarly to the one electron case, let
$|\chi_i\rangle = |\psi_i\rangle\otimes|s_i\rangle$ be orthonormal single electron
states and let $\Psi = \|\chi_1,\dots,\chi_n\rangle$ be their Slater determinant.
Then:

$$
\begin{align*}
\langle \Psi | N^n(\mathbf{r}) | \Psi \rangle &=
\sum_{i=1}^n \langle \chi_i | N^1(\mathbf{r}) | \chi_i \rangle \\
&= \sum_{i=1}^n |\psi_i(\mathbf{r})|^2
\end{align*}
$$

This implies that the expectation value of the $N^n(\mathbf{r})$ for the Slater determinant
is the sum of the probability densities of each of the spatial wavefunctions $\psi_i$.

The density function $\rho$ associated to a general state $\Psi\in\Lambda^n{\mathcal{H}}$
is defined by:

$$
\rho(\mathbf{r}) := \langle \Psi | N^n(\mathbf{r}) | \Psi \rangle
$$


# The Kohn-Sham Equations

Let $\Psi 
Consider an electronic state with density

$$
\rho: \mathbb{R}^3 \rightarrow 
$$
