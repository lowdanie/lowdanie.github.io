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

Intuitively, the $n$-electron local density operator
counts the number of electrons at a position $\mathbf{r}$.

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

## Pair Density

The _double-electron pair local density operator_
$N^1(\mathbf{r}, \mathbf{r}')\in\mathrm{End}(\mathcal{H}^{\otimes 2})$ 
acts as a spatial projector onto a pair of positions
and acts trivially on the spin component:

$$
N^2(\mathbf{r},\mathbf{r}') = |\mathbf{r},\mathbf{r}'\rangle\langle\mathbf{r},\mathbf{r}'|\otimes\mathrm{Id}
$$

The _$n$-electron pair local density operator_ is defined as the symmetric extension
of $N^2(\mathbf{r},\mathbf{r}')$ to $\mathrm{End}(\Lambda^n\mathcal{H})$.

Intuitively $N^2(\mathbf{r},\mathbf{r}')$ counts the number of unique pairs of electrons
at positions $\mathbf{r},\mathbf{r}'\in\mathbb{R}^3$.

The _pair density function_ associated to a state $\|\Psi\rangle\in\Lambda^n\mathcal{H}$
is defined as the expectation value:

$$
\rho_2(\mathbf{r},\mathbf{r}') := \langle\Psi|N^2(\mathbf{r},\mathbf{r}')|\Psi\rangle
$$

# The Reduced Density Matrix

# One Particle RDM

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
of the _$1$-particle reduced density matrix_ (1-RDM)
$\gamma^1\in\mathrm{End}(\mathcal{H})$ associated to
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

## Two Particle RDM

Similarly to the single particle case, if $O^2\in\mathrm{End}(V^{\otimes 2})$ is
a symmetric operator 
and $O^n\in\mathrm{End}(V^{\otimes n})$
is its symmetric extension then, for any anti-symmetric tensor 
$\|v\rangle\in\Lambda^n V$:

$$
\begin{align*}
\langle v | O^n | v \rangle &= \mathrm{Tr}(O^n |v\rangle\langle v|) \\
&= \mathrm{Tr}(O^2 \mathrm{Tr}_{n-2}(\frac{n(n-1)}{2}|v\rangle\langle v|))
\end{align*}
$$

This motivates the definition of the _$2$-particle reduced density matrix_ ($2$-RDM)
associated to a state $\|\Psi\rangle\in\Lambda^n\mathcal{H}$:

$$
\gamma^2 := n(n-1)\mathrm{Tr}_{n-2}(|\Psi\rangle\langle\Psi|)
$$

By the above result, if $O^2\in\mathrm{End}(\mathcal{H}^{\otimes 2})$ is a
symmetric two electron operator, $O^n\in\mathrm{End}(V^{\otimes n})$
is its symmetric extension, $\|\Psi\rangle\in\Lambda^n\mathcal{H}$ is an
$n$-electron state and $\gamma^2$ is its $2$-RDM then:

$$
\langle\Psi | O^n | \Psi\rangle = \frac{1}{2}\mathrm{Tr}(O^2 \gamma^2)
$$

If we apply this to the pair local density operator $N^2(\mathbf{r},\mathbf{r}')$
we can express the pair density function associated to $\Psi$ in terms of
its $2$-RDM:

$$
\begin{align*}
\rho_2(\mathbf{r}_1,\mathbf{r}_2) &= \frac{1}{2}\mathrm{Tr}(N^2(\mathbf{r},\mathbf{r}') \gamma^2) \\
&= \frac{1}{2}\langle\mathbf{r}_1,\mathbf{r}_2| \gamma^2 | \mathbf{r}_1,\mathbf{r}_2\rangle
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
V_{\mathrm{en}}^1 |\psi\rangle\otimes|s\rangle = |v_{\mathrm{en}} \psi \rangle|s\rangle
$$

where $v_{\mathrm{en}}(\mathbf{r})\in L^2(\mathbb{R}^3)$ is the electron-nuclear attraction potential:

$$
v_\mathrm{en}(\mathbf{r}) = -\sum_{i=1}^m \frac{Z_i}{||\mathbf{R}_i - \mathbf{r}||}
$$

Note that by definition:

$$
V_{\mathrm{en}}^1 |\mathbf{r}\rangle = v_{\mathrm{en}}(\mathbf{r})|\mathbf{r}\rangle
$$

In other words, $V_{\mathrm{en}}^1$ is diagonal in the position basis
$\|\mathbf{r}\rangle$ with eigenvalues $v_{\mathrm{en}}(\mathbf{r})$.

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
\langle v_{\mathrm{en}}(\mathbf{r}) | \rho(\mathbf{r}) \rangle_{L^2(\mathbb{R}^3)}
$$

In summary, the expectation value of $V_{\mathrm{en}}^n$ in the state $\|\Psi\rangle$
can be computed in terms of the potential $v_{\mathrm{en}}$ and the
density function associated to $\|\Psi\rangle$.

## Electron Repulsion

The double-electron electron repulsion operator
$V_\mathrm{ee}^2\in\mathrm{End}(\mathcal{H}^{\otimes n})$
is defined in the positional basis by:

$$
V_\mathrm{ee}^2|\mathbf{r}_1\mathbf{r}_2\rangle := \frac{1}{||\mathbf{r}_1 - \mathbf{r}_2||}|\mathbf{r}_1\mathbf{r}_2\rangle 
$$

and acts trivially on the spin component. To facilitate notation we'll define:

$$
v_\mathrm{ee}(\mathbf{r}_1,\mathbf{r}_2) := \frac{1}{||\mathbf{r}_1 - \mathbf{r}_2||}
$$

The $n$-electron electron repulsion operator $V_\mathrm{ee}^2\in\mathrm{End}(\Lambda^n\mathcal{H})$ 
is defined as the symmetric extension of $V_\mathrm{ee}^2$.

Let $\|\Psi\rangle\in\Lambda^n\mathcal{H}$ be an $n$-electron state and $\gamma^2$ its
$2$-RDM. By the previous section:

$$
\langle\Psi|V_\mathrm{ee}^n|\Psi\rangle = \frac{1}{2}\mathrm{Tr}(V_\mathrm{ee}^2\gamma^2)
$$

We'll evaluate the trace in the pair positional basis
$\|\mathbf{r}\_1,\mathbf{r}\_2\rangle$.
By definition, $V\_\mathrm{ee}^2$ is diagonal in that basis with eigenvalues
$v_\mathrm{ee}(\mathbf{r}_1,\mathbf{r}_2)$.
In the previous section we saw that the diagonal elements of $\gamma^2$ in that basis are
given by the pair density function $\rho_2(\mathbf{r}_1,\mathbf{r}_2)$. Together this implies:

$$
\langle\Psi|V_\mathrm{ee}^n|\Psi\rangle = 
\langle v_\mathrm{ee} |\rho_2\rangle_{L^2(\mathbb{R}^3\times\mathbb{R}^3)}
$$

This means that we can express the expectation value of the electron repulsion potential
in terms of the _pair_ density function $\rho(\mathbf{r}_1,\mathbf{r}_2)$.


# Kohn-Sham

Let $\|\Psi\rangle\in\Lambda^n\mathcal{H}$ be an $n$-electron state.

Based on the previous section, we can express it's total expectation energy in terms
of the density function, the pair density function and the $1$-RDM:

$$
\langle\Psi|H^n|\Psi\rangle =
\mathrm{Tr}(T^1\gamma^1) + \langle v_\mathrm{en}|\rho\rangle + \langle v_\mathrm{ee}|\rho_2\rangle
$$

The core difficulty in computing the energy is that the $1$-RDM $\gamma^1$ and the
pair density function $\rho_2$ are high dimensional and complex reductions of the
state $\Psi$. We'd like to express the energy in terms of the simpler density function $\rho$ alone.

The idea behind the Kohn-Sham approach is to approximate the kinetic and electron-repulsion
terms by "non-interacting" versions that depend only on $\rho$. We then collect the error
in a new _exchange correlation_ term $E_{XC}[\rho]$ that also is a functional of $\rho$.

Of course, we do not expect to be able to compute $E_{XC}[\rho]$ precisely but,
since it is only responsible to a relatively small error term, we hope that an approximation
is sufficient.

## The Kohn-Sham Ansatz

To formalize this, we assume that there exist orthonormal orbitals

$$
|\phi_1\rangle,\dots,|\phi_n\rangle\in\mathcal{H}
$$

such that their Slater determinant

$$
\Phi_S = |\phi_1,\dots,\phi_n\rangle
$$

matches the target density $\rho$:

$$
\rho(\mathbf{r}) = \langle \Phi_S | N^n(\mathbf{r}) | \Phi_S\rangle = \sum_{i=1}^n|\phi_i(\mathbf{r})|^2
$$

Note that under this assumption, the density has the form:

$$
\rho(\mathbf{r}) = \sum_{i=1}^n|\phi_i(\mathbf{r})|^2
$$

We can now define the "non-interacting" reference energies and the error term $E_{XC}$.

### Non-Interacting Kinetic Energy

The non-interacting kinetic energy $T_S$ is defined as the expectation kinetic energy for
the Slater determinant $\Psi_S$:

$$
T_S[\{\phi_i\}] := \langle\Psi_S|T^n|\Psi_S\rangle = 
\sum_{i=1}^n\langle\phi_i|T^1|\phi_i\rangle
$$

### Hartree Energy

In order to approximate the expectation electron repulsion energy, we make the
simplifying assumption that the electron densities are uncorrelated.
Specifically, we assume that:

$$
\rho_2(\mathbf{r}_1,\mathbf{r}_2) = \frac{1}{2}\rho(\mathbf{r}_1)\rho(\mathbf{r}_2)
$$

The electron repulsion potential is now given by the Coulomb functional

$$
\begin{align*}
J[\rho] &= \frac{1}{2}\langle v_\mathrm{ee}(\mathbf{r}_1,\mathbf{r}_2) | \rho(\mathbf{r}_1)\rho(\mathbf{r}_2)\rangle \\
&= \frac{1}{2}\sum_{i,j=1}^n\langle\phi_i\phi_j|V_\mathrm{ee}|\phi_i\phi_j\rangle
\end{align*}
$$

### Exchange Correlation Energy

Finally, we define the exchange-correlation functional
$E_{XC}[\rho]$ to be the exact difference between the true energy terms and our
non-interacting approximations:

$$
E_{XC}[\rho] := (T[\rho] - T_S[\{\phi_i\}]) + (\langle v_\mathrm{ee} | \rho_2 \rangle - J[\rho])
$$

### Total Energy

The total energy of the original system with density $\rho$ can now be written as:

$$
E[\rho] = T_S[\{\phi_i\}] + J[\rho] + V_\mathrm{en}[\rho] + E_{XC}[\rho]
$$

Note that under the Kohn-Sham ansatz, $\rho$ is itself a function of the
KS orbitals $\{\phi_i\}$.

In Kohn-Sham DFT, we must first choose an approximation to $E_{XC}[\rho]$. 
Then we use the method of Lagrange multipliers to minimize
$E[\rho]$ with respect to the orbitals $\{\phi_i\}$ subject to the orthonormality
constraint.

## The Kohn-Sham Equation

In the previous section we saw that, under the Kohn-Sham ansatz, the density
is given by:

$$
\rho(\mathbf{r}) = \sum_{i=1}^n|\phi_i(\mathbf{r})|^2
$$

and the total energy is equal to:

$$
E[\rho] = T_S[\{\phi_i\}] + J[\rho] + V_\mathrm{en}[\rho] + E_{XC}[\rho]
$$

In this section we apply the method of Lagrange multipliers to minimize $E$
with respect to the orbitals $\phi_i$ subject to the orthonormality constraint

$$
\langle \phi_i | \phi_j \rangle = \delta_{ij}
$$

First we introduce Lagrange multipliers $\lambda_{ij}$ for $1\leq i,j \leq n$.
We can then setup the Lagrangian

$$
L[\{\phi_i\}] := E[\{\phi_i\}] - \sum_{i,j=1}^n \lambda_{ij}\langle \phi_i | \phi_j \rangle
$$

At solutions to the optimization problem, the variation of $L$ must vanish:

$$
\frac{\delta L}{\delta \phi_i^*} = 0
$$

We'll now derive an expression for the variation and obtain a system of equations
satisfied by a solution $\{\phi_i\}$.

### The Chain Rule

Since most of the terms in the total energy are given in terms of $\rho$,
we'll compute their variations with respect the orbitals using the chain rule.
For this purpose, we'll start by computing the variation of the density:

$$
\begin{align*}
\frac{\delta\rho(\mathbf{r})}{\delta\phi_i^*(\mathbf{r}')}
&= \sum_{j=1}^n 
\frac{\delta}{\delta \phi_i^*(\mathbf{r}')}(\phi_j^*(\mathbf{r})\phi_j(\mathbf{r})) \\
&= \phi_i(\mathbf{r})\delta(\mathbf{r} - \mathbf{r}')
\end{align*}
$$

Therefore, if $F[\rho]$ is any functional of $\rho$ then by the chain rule:

$$
\begin{align*}
\frac{\delta F}{\delta\phi^*(\mathbf{r})} 
&= \left\langle \frac{\delta F}{\delta\rho} \Bigg| \frac{\delta\rho}{\delta\phi_i^*(\mathbf{r})}\right\rangle \\
&= \left\langle \frac{\delta F}{\delta\rho} \Bigg| \phi_i(\mathbf{x})\delta(\mathbf{x}-\mathbf{r})\right\rangle_\mathbf{x} \\
&= \frac{\delta F}{\delta\rho(\mathbf{r})}\phi_i(\mathbf{r})
\end{align*}
$$

### Non-Interacting Kinetic Energy

In the previous section we saw that 

$$
T_S[\{\phi_i\}] = \sum_{i=1}^n\langle\phi_i|T^1|\phi_i\rangle
$$

Therefore:

$$
\begin{align*}
\frac{\delta}{\delta\phi_i^*(\mathbf{r})}T_S
&= \sum_{j=1}^n \frac{\delta}{\delta\phi_i^*(\mathbf{r})}\langle\phi_j|T^1|\phi_j\rangle \\
&= \frac{\delta}{\delta\phi_i^*(\mathbf{r})}\langle\phi_i|T^1|\phi_i\rangle \\
&= T^1\phi_i(\mathbf{r})
\end{align*}
$$

### Hartree-Energy

By definition:

$$
J[\rho] = \frac{1}{2}\langle v_\mathrm{ee}(\mathbf{r}_1,\mathbf{r}_2) | \rho(\mathbf{r}_1)\rho(\mathbf{r}_2)\rangle
$$

Therefore, by the chain rule:

$$
\begin{align*}
\frac{\delta J}{\delta\phi_i^*(\mathbf{r})}
&= \frac{\delta J}{\delta\rho(\mathbf{r})}\phi_i(\mathbf{r}) \\
&= \langle v_\mathrm{ee}(\mathbf{r},\cdot) | \rho\rangle \phi_i(\mathbf{r})
\end{align*}
$$

To facilitate notation, we'll introduce the _Hartree potential_ associated to a
density function $\rho$ denoted by $v_H[\rho]$ and defined by:

$$
v_H[\rho](\mathbf{r}) := \langle v_\mathrm{ee}(\mathbf{r},\cdot) | \rho\rangle
$$

We can now write the functional derivative of $J$ as:

$$
\frac{\delta J}{\delta\phi_i^*} = v_H[\rho]\phi_i
$$

### Electron Nuclear Attraction

We've seen above that the electron-nuclear attraction energy has an expectation
value:

$$
V_\mathrm{en}[\rho] = \langle v_\mathrm{en} | \rho \rangle
$$

Therefore, by the chain rule:

$$
\begin{align*}
\frac{\delta V_\mathrm{en}}{\delta\phi_i^*(\mathbf{r})}
&= \frac{\delta V_\mathrm{en}}{\delta\rho(\mathbf{r})}\phi_i(\mathbf{r}) \\
&= v_\mathrm{en}(\mathbf{r}) \phi_i(\mathbf{r})
\end{align*}
$$

### Exchange Correlation

Since we do not have an explicit definition of $E_{XC}[\rho]$, we'll define the
_exchange correlation potential_ to be the functional derivative of $E_{XC}$:

$$
v_{XC}[\rho](\mathbf{r}) := \frac{\delta E_{XC}}{\delta \rho}
$$

Then by the chain rule:

$$
\frac{\delta E_{XC}}{\delta\phi_i^*(\mathbf{r})} = v_{XC}[\rho](\mathbf{r})\phi_i(\mathbf{r})
$$

### The Kohn-Sham Equation

Putting this all together, the functional derivative of the Lagrangian defined above
is equal to:

$$
\frac{\delta L}{\delta\phi_i^*} = 
(T^1 + v_\mathrm{en} + v_H[\rho] + v_{XC})\phi_i -\sum_{j=1}^n\lambda_ij\phi_j
$$

The Lagrange multiplier condition now becomes:

$$
(T^1 + v_\mathrm{en} + v_H[\rho] + v_{XC}[\rho])\phi_i = \sum_{j=1}^n\lambda_{ij}\phi_j
$$

This motivates the introduction of the _Kohn-Sham Hamiltonian_ $H_{KS}$:

$$
H_{KS}[\rho] := T^1 + v_\mathrm{en} + v_H[\rho] + v_{XC}[\rho]
$$

We can simplify the Lagrange multiplier further by noting that, since
$H_{KS}$ is Hermitian, there is a unitary transformation of the orbitals $\{\phi_i\}$
that diagonalizes $\lambda_{ij}$. Furthermore, unitary transformations of the orbitals
do not change the corresponding density $\rho$. Therefore, without loss of generality
we can assume that the orbitals and Lagrange multipliers satisfy:

$$
H_{KS}[\rho]\phi_i = \lambda_i\phi_i
$$

Since $H_{KS}$ depends on the density $\rho$ which in turn depends on the orbitals
$\phi$, this must be solved via a self-consistent field method similarly to the
Hartree-Fock equation.

## The Matrix Equation

In this section we assume that the Kohn-Sham orbitals $\{\phi_i\}$ can be expanded
as a linear combination of a fixed set of $b$ basis functions $\{\eta_i}_{i=1}^b$.
Under this assumption,
the Kohn-Sham equation can be expressed as a matrix equation which parallels the
Roothan equations in Hartree-Fock theory.

First, we'll collect the linear coefficients of the Kohn-Sham orbitals into the columns of the
_coefficient matrix_
$\mathbf{C}\in\mathrm{Mat}_{b\times n}(\mathbb{C})$:

$$
|\phi_i\rangle = \sum_{j=1}^bC_{ji}|\eta_j\rangle
$$

Similarly to the Roothaan equations, we also define the overlap matrix $\mathbf{S}\in\mathrm{Mat}_{b\times b}(\mathbb{C})$:

$$
S_{ij} := \langle\eta_i|\eta_i\rangle
$$

and the density matrix $\mathbf{P}\in\mathrm{Mat}_{b\times b}(\mathbb{C})$:

$$
\mathbf{P}=\mathbf{C}\mathbf{C}^*
$$

Note that the density function $\rho$ can be expressed in terms of the density matrix via:

$$
\rho = \sum_{i,j=1}^b P_{ij}\eta_i\eta_j^*
$$

Finally, we define the _Kohn-Sham matrix_ $\mathbf{F}^{KS}\in\mathrm{Mat}\_{b\times b}(\mathbb{C})$
to contain the matrix elements of the Kohn-Sham Hamiltonian:

$$
F_{ij}^{KS}[\mathbf{P}] := \langle\eta_i | H_{KS}[\rho] | \eta_j\rangle
$$

Analogously to the Roothaan equation, the Kohn-Sham equation can now be expressed as:

$$
\mathbf{F}^{KS}[\mathbf{P}]\mathbf{C} = \mathbf{S}\mathbf{C}\mathbf{D}
$$

This is a generalized eigenvalue equation which again can be solved identically to the
Roothaan equation. All that remains is to compute the Kohn-Sham matrix $\mathbf{F}^{KS}[\mathbf{P}]$.

### The Kohn-Sham Matrix

By the definition of the Kohn-Sham Hamiltonian, the Kohn-Sham matrix is a sum of the following terms:

$$
\mathbf{F}^{KS}[\mathbf{P}] = \mathbf{H}^\mathrm{core} + \mathbf{J}[\mathbf{P}] + \mathbf{V}^{XC}[\mathbf{P}]
$$

where $\mathbf{H}^\mathrm{core}$ contains the non-interacting kinetic energy and
the electron-nuclear attraction similarly to Hartree-Fock.

We'll now compute the Hartree matrix $\mathbf{J}[\mathbf{P}]$. By definition:

$$
\mathbf{J}[\mathbf{P}]_{ij} = \langle\eta_i | v_H[\rho] | \eta_j \rangle
$$

Using the above expansion of $\rho$ in terms of the density matrix $\mathbf{P}$:

$$
 v_H[\rho](\mathbf{r}) = \sum_{kl}P_{kl} \langle \eta_l | v_\mathrm{ee}(\mathbf{r},\cdot) | \eta_k \rangle
$$

Plugging this into the definition of the Hartree matrix:

$$
\mathbf{J}[\mathbf{P}]_{ij} = \sum_{kl} P_{kl} \langle \eta_i \eta_l | V_\mathrm{ee} | \eta_j \eta_k \rangle
$$

Finally, the exchange correlation matrix is by definition equal to:

$$
V_{ij}^{XC}[\mathbf{P}] = \langle \eta_i | v_{XC}[\rho] | \eta_j \rangle
$$

Unlike the Hartree matrix, the exchange correlation matrix cannot in general be reduced to
standard electron integrals and must be computed numerically via a evaluation on a grid.

### Restricted Kohn-Sham

Similarly to restricted Hartree-Fock, in restricted Kohn-Sham (RKS) we assume that
the Kohn-Sham orbitals form a closed-shell system. In this case, the $n$ spin orbitals
are replaced with $n/2$ spatial orbitals $\{\phi_i}_{i=1}^{n/2}$. The density matrix must
by modified to account for the fact that each spatial orbital appears in two spin orbitals:

$$
\mathbf{P} = 2\mathbf{C}\mathbf{C}^*
$$



