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

# The Reduced Density Matrix

In the previous section we defined the $n$-electron local density operator $N^n(\mathbf{r})$
as the symmetric extension of $N^1(\mathbf{r})$. 

In general, if $O^1\in\mathrm{End}(V)$ is an operator and $O^n\in\mathrm{End}(V^{\otimes n})$
is its symmetric extension then, for any anti-symmetric tensor $\|v\rangle\in\Lambda^n V$:

$$
\begin{align*}
\langle v | O^n | v \rangle &= \mathrm{Tr}(O^n |v\rangle\langle v|) \\
&= \mathrm{Tr}(O^1 \mathrm{Tr}_{n-1}(n|v\rangle\langle v|))
\end{align*}
$$

Where

$$
\mathrm{Tr}_{n-1}: \mathrm{End}(V^{\otimes n}) \rightarrow \mathrm{End}(V)
$$

is the partial trace on the last $n-1$ factors. This motivates the introduction 
of the _reduced density matrix_ $\gamma^1\in\mathrm{End}(\mathcal{H})$ associated to
an $n$-electron state $|\Psi\rangle\in\Lambda^n\mathcal{H}$ defined by:

$$
\gamma^1 := \mathrm{Tr}_{n-1}(n|\Psi\rangle\langle\Psi|)
$$

Note that in the single-electron case, the reduced density matrix associated to
$\|\chi\rangle\in\mathcal{H}$ is simply the density operator
$\|\chi\rangle\langle\chi\|$.

By the above result, it immediately follows that if $O^1\in\mathrm{End}(\mathcal{H})$
is a single-electron operator, $O^n\in\mathrm{End}(\Lambda^n\mathcal{H})$ is its
symmetric extension, $|\Psi\rangle\in\Lambda^n\mathcal{H}$ is an $n$-electron state
and $\gamma^1$ is its reduced density matrix then:

$$
\langle \Psi | O^n | \Psi \rangle = \mathrm{Tr}(O^1\gamma^1)
$$

As a special case, the density function associated to a state $\|\Psi\rangle$
contains the diagonal elements of $\gamma^1$ in the basis $\|\mathbf{r}\rangle$:

$$
\begin{align*}
\rho(\mathbf{r}) &= \langle \Psi | N^n(\mathbf{r}) | \Psi \rangle \\
&= \mathrm{Tr}(N^1(\mathbf{r})\gamma^1) \\
&= \mathrm{Tr}(|\mathbf{r}\rangle\langle\mathbf{r}|\gamma^1) \\
&= \langle \mathbf{r} | \gamma^1 | \mathbf{r} \rangle
\end{align*}
$$

# Expectation Values

A key observation of density functional theory is that the expectation values of
some of the operators that appear in the electronic Hamiltonian can be expressed
in terms of density functions.

## Electron Nuclear Attraction

First consider the electron-nuclear attraction operator. In the single-electron
case, the operator $V_{\mathrm{en}}^1\in\mathrm{End}(\mathcal{H})$ is defined by

$$
V_{\mathrm{en}}^1 |\psi\rangle\otimes|s\rangle = |v_{\mathrm{ne}} \psi \rangle|s\rangle
$$

where $v_{\mathrm{ne}}(\mathbf{r})\in L^2(\mathbb{R}^3)$ is the electron-nuclear attraction potential:

$$
v(\mathbf{r}) = -\sum_{i=1}^m \frac{Z_i}{||\mathbf{R}_i - \mathbf{r}||}
$$

Note that by definition:

$$
V_{\mathrm{en}}^1 |\mathbf{r}\rangle = v_{\mathrm{ne}}(\mathbf{r})|\mathbf{r}\rangle
$$

In other words, $V_{\mathrm{en}}^1$ is diagonal in the position basis
$\|\mathbf{r}\rangle$ with eigenvalues $v_{\mathrm{ne}}(\mathbf{r})$.

The $n$-electron electron-nuclear attraction operator
$V_{\mathrm{en}}^n\in\mathrm{End}(\Lambda^n\mathcal{H})$ is defined to be the symmetric
extension of $V^1$.

Now let $\|\Psi\rangle\in\Lambda^n\mathcal{H}$ be an $n$-electron state and let $\gamma^1$
be its reduced density matrix. By the previous section, the expectation
value of $V_{\mathrm{en}}^n$ with respect to $\|\Psi\rangle$ is given by

$$
\langle \Psi | V_{\mathrm{en}}^n | \Psi \rangle =
\mathrm{Tr}(V_{\mathrm{en}}^1 \gamma^1)
$$

Since $V_{\mathrm{en}}^1$ is diagonal in the positional basis $\|\mathbf{r}\rangle$,
the trace is equal to the inner product of the eigenvalues of $V_{\mathrm{en}}^1$
and the diagonal elements of $\gamma^1$ in that basis.
But in the previous section we saw that by the diagonal elements of $\gamma^1$ are given
by the density function:

$$
\langle\mathbf{r}|\gamma^1|\mathbf{r}\rangle = \rho(\mathbf{r})
$$

Together this implies that:

$$
\langle \Psi | V_{\mathrm{en}}^n | \Psi \rangle =
\langle v_{\mathrm{ne}}(\mathbf{r}) | \rho(\mathbf{r}) \rangle_{L^2(\mathbb{R}^3)}
$$

In summary, the expectation value of $V_{\mathrm{en}}^n$ in the state $\|\Psi\rangle$
can be computed in terms of the potential $v_{\mathrm{ne}}$ and the
density function associated to $\|\Psi\rangle$.


# The Kohn-Sham Equations

Let $\Psi 
Consider an electronic state with density

$$
\rho: \mathbb{R}^3 \rightarrow 
$$
