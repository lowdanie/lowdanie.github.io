---
layout: post
title: "Electron Integrals"
date: 2025-07-26
mathjax: true
utterance-issue: 13
---

# The Electronic Hamiltonian 

The goal of this post is to determine the ground state of a molecule from first 
principles.

In this section we'll formalize our notion of the state space of a molecule. 
We'll then parameterize a subset of the state space using Slater determinants. 
Finally, we'll apply the
[variational method](https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics))
to derive a function of the parameters which has a minimum at the 
Slater determinant that best approximates the ground state.

## Molecular Orbitals

The state of a single electron has two components, one related to position in space
and the other a quantum analog of angular momentum called
[spin](https://en.wikipedia.org/wiki/Spin_(physics)).

The position is characterized by a [wave function](https://en.wikipedia.org/wiki/Wave_function)
 which is a complex valued
[square integrable](https://en.wikipedia.org/wiki/Square-integrable_function)
function on $\mathbb{R}^3$:

$$
\Psi(\mathbf{r}) \in L^2(\mathbb{R}^3)
$$

The space of [square integrable](https://en.wikipedia.org/wiki/Square-integrable_function)
complex valued functions on $\mathbb{R}^3$, $L^2(\mathbb{R}^3)$, is a 
[Hilbert space](https://en.wikipedia.org/wiki/Hilbert_space)
where the inner product of two functions $\phi,\psi\in L^2(\mathbb{R}^3)$ is defined
by:

$$
\langle \phi | \psi \rangle := 
\int_{\mathbb{R}^3}\phi^*(\mathbf{r})\psi(\mathbf{r})d\mathbf{r}
$$

The spin of an electron is a complex linear combination of two states called
_spin up_ and _spin down_. The spin component of the electron state can therefore be
identified with the vector space $\mathbb{C}^2$.

Combining position and spin, 
the state space of an electron, denoted $\mathcal{H}$, is equal to the tensor product:

$$
\mathcal{H} := L^2(\mathbb{R}^3)\otimes\mathbb{C}^2
$$

The space $\mathcal{H}$ is also a Hilbert space since its a tensor product of 
two Hilbert spaces. Following standard quantum mechanics terminology, we'll use the
terms _state space_ and _Hilbert space_ interchangeably.

In order to extend this to $n$ electrons, we'll first need to define
the notion of an
[anti-symmetric tensor](https://en.wikipedia.org/wiki/Exterior_algebra#Alternating_tensor_algebra).

> **Definition (Symmetric Group Action).**
> Let $V$ be a vector space and $n\in\mathbb{Z}$ a positive integer.
> Let $S_n$ denote the symmetric group on $n$ elements.
>
> For each permutation $\sigma\in S_n$, define 
> $P_\sigma\in\mathrm{Aut}(V^{\otimes n})$
> to be the linear automorphism of $V^{\otimes n}$ defined by:
>
> $$
> \begin{align*}
> P_\sigma: V^{\otimes n} &\rightarrow V^{\otimes n} \\
> |v_1\dots v_N\rangle &\mapsto |v_{\sigma^{-1}(1)}\dots v_{\sigma^{-1}(n)}\rangle
> \end{align*}
> $$

We'll use the notation $\mathrm{sgn}(\sigma)$ denote the 
[sign](https://en.wikipedia.org/wiki/Parity_of_a_permutation) of a permutation
$\sigma\in S_n$.

> **Definition (Anti-symmetric Tensor).**
> Let $V$ be a vector space and $n\in\mathbb{Z}$ a positive integer.
>
> A tensor $|\Psi\rangle\in V^{\otimes n}$ is defined to be _anti-symmetric_ if, 
> for all $\sigma\in S_n$:
>
> $$
> P_\sigma |\Psi\rangle = \mathrm{sgn}(\sigma)|\Psi\rangle
> $$

Since $P_\sigma$ is a linear transformation of $V^{\otimes n}$ for all $\sigma\in S_n$,
it follows that the set of anti-symmetric tensors is a subspace of $V^{\otimes n}$.
This subspace is called the _exterior product_ as formalized in the following definition.

> **Definition (Exterior Product).**
> Let $V$ be a vector space and $n\in\mathbb{Z}$ a positive integer.
>
> The _$n$-th exterior product_ of $V$, denoted $\Lambda^n V \subset V^{\otimes n}$
> is defined to be the subspace of anti-symmetric tensors in $ V^{\otimes n}$.

According to the
[Pauli exclusion principle](https://en.wikipedia.org/wiki/Pauli_exclusion_principle),
the Hilbert space of a collection of $n$ electrons is equal to $\Lambda^n\mathcal{H}$
I.e, an $n$ electron state is an anti-symmetric tensor of $n$ single electron states.

Now consider a molecule with $n$ electrons and $m$ nuclei.
Since nuclei are orders of magnitude heavier than electrons, 
it's common in quantum chemistry to use the
[Born Oppenheimer](https://en.wikipedia.org/wiki/Born%E2%80%93Oppenheimer_approximation)
approximation which assumes that the nuclei are stationary point masses.

This means that quantum state of the molecule is equal to the Hilbert space of its
$n$ electrons. In quantum chemistry, elements of $\Lambda^n\mathcal{H}$
are referred to as
[molecular orbitals](https://en.wikipedia.org/wiki/Molecular_orbital).

## Slater Determinants

In the previous section we saw that the joint state space of $n$ electrons is equal
to the space of alternating tensors
$\Lambda^n\mathcal{H} \subset \mathcal{H}^{\otimes n}$.

Let $V$ be a vector space.
The goal of this section is to show how to construct an alternating tensor in
$\Lambda^n V$ from an arbitrary tensor $|\Phi\rangle\in V^{\otimes n}$.

First we'll consider the case where $n=2$. Consider the decomposable tensor:

$$
|v_1 v_2\rangle \in V^{\otimes 2}
$$

In general, $\|v_1 v_2\rangle$ is not in general anti-symmetric. To see why,
consider the permutation $\sigma=(1,2)\in S_2$. Then

$$
P_\sigma |v_1 v_2\rangle = |v_2 v_1\rangle
$$

Furthermore, $\mathrm{sgn}(\sigma) = -1$ and so

$$
\mathrm{sgn}(\sigma)|v_1 v_2\rangle = -|v_1 v_2\rangle
$$

Therefore, in general:

$$
P_\sigma |v_1 v_2\rangle \neq \mathrm{sgn}(\sigma)|v_2 v_1\rangle
$$

We can fix this by adding the missing term $-\|v_2 v_1\rangle$ to our state
and defining:

$$
|\Psi\rangle = |v_1 v_2\rangle - |v_2 v_1\rangle
$$

Now when we apply $P_\sigma$ we get:

$$
\begin{align*}
P_\sigma(|\Psi\rangle) &= P_\sigma |v_1 v_2\rangle - P_\sigma |v_2 v_1\rangle \\
&= |v_2 v_1\rangle - |v_1 v_2\rangle \\
&= -|\Psi\rangle
\end{align*}
$$

This implies that $\|\Psi\rangle$ is an anti-symmetric tensor.

We can extend this construction to the case of $n$ electrons using the
[antisymmetrization map](https://en.wikipedia.org/wiki/Exterior_algebra#Alternating_tensor_algebra)
$\mathcal{A}^n$ which maps the tensor product $V^{\otimes n}$ to the space of
alternating tensors $\Lambda^n V$.

> **Definition (Antisymmetrization).**
> Let $V$ be a vector space and $n\in\mathbb{Z}$ a positive integer.
>
> The _antisymmetrization map_, denoted $\mathrm{Alt}^n$, is defined as:
>
> $$
> \begin{align*}
> \mathrm{Alt}^n: V^{\otimes n} &\rightarrow V^{\otimes n} \\
> |\phi\rangle &\mapsto \sum_{\sigma\in S_n}\mathrm{sgn}(\sigma) P_\sigma |\phi\rangle
> \end{align*}
> $$

> **Claim (Antisymmetrization).**
> Let $V$ be a vector space and $n\in\mathbb{Z}$ a positive integer.
>
> Then, for all tensors $\|\phi\rangle\in V^{\otimes n}$,
> $\mathrm{Alt}^n(\|\phi\rangle)$ is an anti-symmetric tensor.
>
> In particular, the image of the antisymmetrization map is contained in $\Lambda^n V$:
>
> $$
> \mathrm{Alt}^n : V^{\otimes n} \rightarrow \Lambda^n V
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

Let  $\|\phi\rangle\in V^{\otimes n}$ be a tensor. To show that 
$\mathrm{Alt}^n(\|\phi\rangle)$ is anti-symmetric we must show that for all $\sigma\in S_n$,

$$
P_\sigma \mathrm{Alt}^n(|\phi\rangle) = \mathrm{sgn}(\sigma)\mathrm{Alt}^n(|\phi\rangle).
$$

Let $\sigma\in S_n$ be a permutation. Then, by the definition of the antisymmetrization map:

$$
\begin{align*}
P_\sigma \mathrm{Alt}^n(|\phi\rangle) &=
P_\sigma \sum_{\sigma'\in S_n}\mathrm{sgn}(\sigma') P_{\sigma'} |\phi\rangle \\
&= \sum_{\sigma'\in S_n}\mathrm{sgn}(\sigma)^2\mathrm{sgn}(\sigma') P_{\sigma\sigma'} |\phi\rangle \\
&= \mathrm{sgn}(\sigma)\sum_{\sigma'\in S_n}\mathrm{sgn}(\sigma\sigma') P_{\sigma\sigma'} |\phi\rangle \\
&= \mathrm{sgn}(\sigma) \mathrm{Alt}^n(|\phi\rangle)
\end{align*}
$$

The last equality follows from the fact that composition with $\sigma$ is an automorphism
of $S_n$.

</div>
</details>

The [Slater determinant](https://en.wikipedia.org/wiki/Slater_determinant)
is a normalized version of the antisymmetrization map.

> **Definition (Slater Determinant).**
> Let $V$ be a vector space and $n\in\mathbb{Z}$ a positive integer.
> 
> The _Slater determinant_ of a decomposable tensor
> $\|v_1\dots v_n\rangle\in V^{\otimes n}$ is denoted by
> $\|v_1,\dots,v_n\rangle$ (note the commas) and defined by:
>
> $$
> |v_1,\dots,v_n\rangle := \frac{1}{\sqrt{n!}}\mathrm{Alt}^n(|v_1\dots v_n\rangle) \in \Lambda^n V
> $$

In practice, all anti-symmetric tensors considered in this post will be Slater determinants.

## Molecular Operators

In quantum mechanics, [observables](https://en.wikipedia.org/wiki/Observable), including
energy, correspond to [self-adjoint operators](https://en.wikipedia.org/wiki/Self-adjoint_operators)
of Hilbert space. We'll be using the terms _operator_ and _linear transformation_
interchangeably.

In particular, the energy of an $n$-electron molecule corresponds to a self-adjoint operator
on $\Lambda^n\mathcal{H}$.

Let $V$ be a Hilbert space. In this section we'll show how to construct
self-adjoint operators on $\Lambda^n V$ from self-adjoint operators on $\Lambda^k V$
for $1 \leq k \leq n$.

Rather than directly constructing linear transformations on $\Lambda^n V$, we'll
instead construct linear transformations of $V^{\otimes n}$ that commute with 
the action of $S_n$ as formalized in the following definition.

> **Definition (Symmetric Operator).**
> Let $V$ be a vector space and $n\in\mathbb{Z}$ a positive integer.
> An operator $T\in\mathrm{End}(V^{\otimes n})$ is defined to be _symmetric_ if it commutes
> with the $S_n$ action on $V^{\otimes n}$.
>
> In other words, $T$ is symmetric if for all $\sigma\in S_n$:
>
> $$
> P_\sigma T = T P_\sigma
> $$

In our context, the significance of symmetric operators on $V^{\otimes n}$
is that they can be restricted to operators on $\Lambda^n V$.

> **Claim (Symmetric Restriction).**
> Let $V$ be a vector space, $n\in\mathbb{Z}$ a positive integer
> and $T\in\mathrm{End}(V^{\otimes n})$ a symmetric operator on $V^{\otimes n}$.
>
> Then, for any anti-symmetric tensor $|\Psi\rangle\in V^{\otimes n}$,
> $T|\Psi\rangle$ is also anti-symmetric. In particular, $T$ can be restricted
> to an operator on $\Lambda^n V$.

By claim XXX, in order to construct an operator on $\Lambda^n V$ it's sufficient to
construct a symmetric operator on $V^{\otimes n}$.
We'll now see how to build symmetric operators on $V^{\otimes n}$ 
from arbitrary operators on $V^{\otimes k}$ where $k \leq n$.

> **Definition (Symmetric Extension).**
> Let $V$ be a vector space, $k \leq n\in\mathbb{Z}$ positive integers
> and $T^k\in\mathrm{End}(V^{\otimes k})$ an operator on $V^{\otimes k}$.
>
> Define the operator $T_k^n\in\mathrm{End}(V^{\otimes n})$ to act as $T^k$
> on the first $k$ factors of $V^{\otimes n}$ and as the identity on the remaining
> factors:
>
> $$
> T_k^n|v_1\dots v_n\rangle := 
> (T^k|v_1\dots v_k\rangle)|v_{k+1}\dots v_n\rangle
> $$
>
> The _symmetric extension_ of $T^k$, $T^n\in\mathrm{End}(V^{\otimes n})$, is defined
> to be:
>
> $$
> T^n := \frac{1}{k!(n-k)!}\sum_{\sigma\in S_n} P_{\sigma^{-1}} T_k^n P_{\sigma}
> $$ 

> **Claim (Symmetric Extension).**
> Let $V$ be a vector space, $k \leq n\in\mathbb{Z}$ positive integers
> and $T^k\in\mathrm{End}(V^{\otimes k})$ an operator on $V^{\otimes k}$.
>
> Then, the symmetrization extension of $T^k$, $T^n\in\mathrm{End}(V^{\otimes n})$,
> is a symmetric operator on $V^{\otimes n}$. In addition, if $T^k$
> is self-adjoint then $T^n$ is self-adjoint.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

First we'll show that $T^n$ is symmetric. Let $\sigma\in S_n$ be a permutation.

Then, by the definition of $T^n$:

$$
\begin{align*}
P_{\sigma^{-1}} T^n P_{\sigma} 
&= P_{\sigma^{-1}} \left( \frac{1}{k!(n-k)!}\sum_{\sigma'\in S_n} P_{\sigma'^{-1}} T_k^n P_{\sigma'}\right) P_{\sigma} \\
&=  \frac{1}{k!(n-k)!}\sum_{\sigma'\in S_n} P_{(\sigma'\sigma)^{-1}} T_k^n P_{\sigma'\sigma} \\
&= T^n
\end{align*}
$$

The last equality follows from the fact that $\sigma^{-1}$ is an automorphism
of size $k$ subsets of $\\{1,\dots,n\\}$.

Now suppose that $T^k$ is self adjoint, i.e, $(T^k)^* = T^k$. It's easy to see
that $T_k^n$ is self adjoint at well. In addition, $P_\sigma$ is unitary for all
permutations $\sigma\in S_n$ which implies that:

$$
\begin{align*}
(T_k^n)^*
&= (P_{\sigma^{-1}} T_k^n P_\sigma)^* \\
&= P_\sigma^* (T_k^n)^* P_{\sigma^{-1}}^* \\
&= P_{\sigma^{-1}} T_k^n P_\sigma \\
&= T_k^n
\end{align*}
$$

This means that $T^n$ is a sum of self-adjoint operators which implies
that it is self-adjoint.

</div>
</details>

An important notion in quantum mechanics is the 
[expectation value](https://en.wikipedia.org/wiki/Expectation_value_(quantum_mechanics))
of an operator in a given state:

> **Definition (Expectation Value).**
> Let $V$ be a Hilbert space.
>
> Let $\|\psi\rangle\in V$ be a vector and
> $T\in\mathrm{End}(V)$ an operator.
>
> The _expectation value_ of $T$ in state $\|\psi\rangle$ is defined to be:
>
> $$
> \langle \psi | T | \psi \rangle
> $$

Let $V$ be a Hilbert space and $n\in\mathbb{Z}$ a positive integer.

In section XXX we saw how to construct elements of 
$\Lambda^n V$ out of $n$ vectors $\|v_1\rangle,\dots,\|v_n\rangle\in V$
using a Slater determinant:

$$
|v_1,\dots,v_n\rangle\in\Lambda^n V
$$

Similarly, saw how to use [symmetric extension](XXX)
to construct an operator $T^n\in\mathrm{End}(\Lambda^n V)$ out of an operator
$T^k\in\mathrm{End}(V^{\otimes k})$.

We'll now derive a formula for the expectation value $T^n$ in state
$\|v_1,\dots,v_n\rangle$ in terms of the constituent operator $T^k$ and states
$\|v_1\rangle,\dots,\|v_n\rangle$.

First let's introduce some convenient notation for dealing with sequences of integers.
For a given integer $n$, we'll use the standard notation:

$$
[n] := \{1,\dots,n\}
$$

Given another integer $k$, we'll denote the set of length $k$ sequences
of integers in $[n]$ by $[n]^k$.

Finally, given $n$ vectors $v_1,\dots,v_n\in V$ be vectors and a seqeuence
$I = (i_1,\dots,i_k) \in [n]^k$ we'll define:

$$
|v_I\rangle := |v_{i_1}\dots v_{i_k}\rangle \in V^{\otimes k}
$$

> **Claim (Slater Determinant Expectation).**
> Let $V$ be an inner-product space and $k \leq n\in\mathbb{Z}$ positive integers.
>
> Let $v_1,\dots,v_n\in V$ be orthonormal vectors in $V$.
> Let $T^k\in\mathrm{End}(V^{\otimes k})$ be a symmetric operator on $V^{\otimes k}$
> and $T^n\in\mathrm{End}(V^{\otimes n})$ its symmetric extension.
>
> Then:
>
> $$
> \langle v_1,\dots,v_n | T^n | v_1,\dots,v_n \rangle =
> \frac{1}{k!}\sum_{ I\in [n]^k } \sum_{\sigma\in S_k}
> \mathrm{sgn}(\sigma) \langle v_I | T^k P_\sigma | v_I \rangle
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

By definition XXX,

$$
T^n = \frac{1}{k!(n-k)!}\sum_{ \rho \in S_n }
P_{\rho^{-1}} T_k^n P_{\rho}
$$

And by definition XXX:

$$
|v_1,\dots,v_n\rangle = 
\frac{1}{\sqrt{n!}} 
\sum_{\sigma \in S_n}\mathrm{sgn}(\sigma)
P_{\sigma}|v_1 \dots v_n\rangle
$$

Therefore,

$$
\begin{align*}
\langle v_1,\dots,v_n | T^n | v_1,\dots,v_n\rangle &=
\frac{1}{k!(n-k)!n!}
\sum_{\rho\in S_n}
\sum_{\sigma,\sigma'\in S_n}
\mathrm{sgn}(\sigma)\mathrm{sgn}(\sigma')
\langle v_1\dots v_n | 
P_{\sigma^{-1}} P_{\rho^{-1}} T_k^n P_{\rho} P_{\sigma'}
| v_1 \dots v_n \rangle \\
&= \frac{1}{k!(n-k)!n!}
\sum_{\rho\in S_n}
\sum_{\sigma,\sigma'\in S_n}
\mathrm{sgn}(\sigma)\mathrm{sgn}(\sigma')
\langle v_1\dots v_n | 
P_{(\rho\sigma)^{-1}} T_k^n P_{\rho\sigma'}
| v_1 \dots v_n \rangle \\
&= \frac{1}{k!(n-k)!}
\sum_{\sigma,\sigma'\in S_n}
\mathrm{sgn}(\sigma)\mathrm{sgn}(\sigma')
\langle v_1\dots v_n | 
P_{\sigma^{-1}} T_k^n P_{\sigma'}
| v_1 \dots v_n \rangle
\end{align*}
$$

Now, let $\sigma,\sigma'\in S_n$ be permutations, and define the sequences 
$K :=(1,\dots,k)\in [n]^k$ and $K^c := (k+1,\dots,n)\in [n]^{n-k}$.

Since $T_k^n$ only acts on the first $k$ factors of $V^{\otimes n}$:

$$
\langle v_1\dots v_n | 
P_{\sigma^{-1}} T_k^n P_{\sigma'}
| v_1 \dots v_n \rangle =
\langle v_{\sigma^{-1}(K)} | T^k | v_{\sigma'^{-1}(K)} \rangle 
\langle v_{\sigma^{-1}(K^c)} | v_{\sigma'^{-1}(K^c)} \rangle
$$

By the orthonormality of $v_1,\dots,v_n$, 

$$
\langle v_{\sigma^{-1}(K^c)} | v_{\sigma'^{-1}(K^c)} \rangle =
\begin{cases}
1 & \sigma'^{-1}(K^c) = \sigma^{-1}(K^c) \\
0 & \mathrm{else}
\end{cases}
$$

We'll now consider the non-zero case where $\sigma'^{-1}(K^c) = \sigma^{-1}(K^c)$.

First, note that in this case $\sigma^{-1}(K)$ and $\sigma'^{-1}(K)$ have the
same elements. Therefore, there is a unique permutation $\tau\in S_k$ such that:

$$
P_\tau |v_{\sigma'^{-1}(K)}\rangle = |v_{\sigma^{-1}(K)}\rangle
$$

Note that since $\sigma'^{-1}(K^c) = \sigma^{-1}(K^c)$:

$$
\mathrm{sgn}(\sigma') = \mathrm{sgn}(\sigma)\mathrm{sgn}(\tau)
$$

Which implies that

$$
\mathrm{sgn}(\sigma)\mathrm{sgn}(\sigma') = \mathrm{sgn}(\tau)
$$

Plugging this all back into XXX we get:

$$
\begin{align*}
\langle v_1,\dots,v_n | T^n | v_1,\dots,v_n\rangle
&= \frac{1}{k!(n-k)!}
\sum_{\sigma \in S_n}
\sum_{\tau\in S_k}
\mathrm{sgn}(\tau)
\langle v_{\sigma^{-1}(K)} | T^k P_{\tau}| v_{\sigma^{-1}(K)} \rangle \\
\end{align*}
$$

Let $P_{n,k}\subset [n]^k$ denote the set of sequences in $[n]^k$ that have unique
elements. Note that $\sigma^{-1}(K)$ is in $P_{n,k}$ and that as $\sigma$ ranges over all permutations
in $S_n$, each element of $P_{n,k}$ appears $(n-k)!$ times. Therefore, we can rewrite XXX
as:

$$
\begin{align*}
\langle v_1,\dots,v_n | T^n | v_1,\dots,v_n\rangle
&= \frac{1}{k!}
\sum_{I \in P_{n,k}}
\sum_{\tau\in S_k}
\mathrm{sgn}(\tau)
\langle v_{I} | T^k P_{\tau}| v_{I} \rangle \\
\end{align*}
$$

To conclude the proof, we'll show that the sum in XXX can be replaced with a sum over _all_
sequences in $[n]^k$ rather than just the ones with unique elements.

Let $I\in [n]^k$ be an arbitrary sequence. We claim that if the elements of
$I$ are not unique then

$$
\sum_{\tau\in S_k}
\mathrm{sgn}(\tau)
\langle v_I | T^k P_{\tau}| v_I \rangle = 0
$$

To prove this, suppose that there exists $i\neq j \in [n]$ such that $I_i = I_j$
and let $(i,j)\in S_n$ denote the permutation that swaps $i$ and $j$.

Then

$$
\begin{align*}
\sum_{\tau\in S_k}
\mathrm{sgn}(\tau)
\langle v_I | T^k P_\tau| v_I \rangle
&= \sum_{\tau\in S_k}
\mathrm{sgn}((\tau(i,j))
\langle v_I | T^k P_{\tau(i,j)}| v_I \rangle \\
&= -\sum_{\tau\in S_k}
\mathrm{sgn}((\tau)
\langle v_I | T^k P_\tau P_{(i,j)}| v_I \rangle \\
&= -\sum_{\tau\in S_k}
\mathrm{sgn}((\tau)
\langle v_I | T^k P_\tau| v_I \rangle
\end{align*}
$$

which implies that

$$
\sum_{\tau\in S_k}
\mathrm{sgn}(\tau)
\langle v_I | T^k P_{\tau}| v_I \rangle = 0
$$

_q.e.d_

</div>
</details>

## Closed Shell

Recall that the Hilbert space $\mathcal{H}$ of single electron space is defined as:

$$
\mathcal{H} := L^2(\mathbf{R}^3)\otimes \mathbb{C}^2
$$

Where the seconds factor $\mathbb{C}^2$ represents the two-dimensional spin space
spanned by spin down, denoted $\|0\rangle$ and spin up, denoted $\|1\rangle$.

Therefore, the decomposable single electron states $\|\chi\rangle\in\mathcal{H}$ are of the form:

$$
|\chi\rangle = |\psi\rangle |s\rangle
$$

where $\psi\in L^2(\mathbf{R}^3)$ is a positional wave function and $s\in\\{0,1\\}$
indicates if the spin state is up or down.

Until now we've been considering $n$-electron Slater determinants of arbitrary sequences of
single-electron states $\|\chi_1\rangle,\dots,\|\chi_n\rangle\in\mathcal{H}$.
In general, molecules are most stable when, for each positional wave function
$\|\psi\rangle$, either both or neither of the possible spin states are occupied.
This type of electron configuration is called
[closed shell](https://en.wikipedia.org/wiki/Electron_configuration#Open_and_closed_shells)
as formalized in the following definition.

> **Definition (Closed Shell Slater Determinant).**
> Let $n\in\mathbb{Z}$ be a positive integer and 
> $\psi_1,\dots,\psi_n\in L^2(\mathbb{R}^3)$ positional wave functions.
>
> The _closed shell Slater determinant_ with positions $\psi_1,\dots,\psi_n$
> is denoted $\|\psi_1,\dots,\psi_n\rangle\in\Lambda^{2n}(\mathcal{H})$ 
> and defined to be the Slater determinant of
> the $2n$ single-electron states 
>
> $$
> |\psi_1\rangle|0\rangle,|\psi_1\rangle|1\rangle,\dots,
> |\psi_n\rangle|0\rangle,|\psi_n\rangle|1\rangle \in \mathcal{H}
> $$

In other words, the closed shell Slater determinant of
$\psi_1,\dots,\psi_n\in L^2(\mathbb{R}^3)$ is given by:

$$
|\psi_1,\dots,\psi_n\rangle := 
|\psi_10,\psi_11,\dots,\psi_n0\psi_n1\rangle\in\Lambda^{2n}(\mathcal{H})
$$

From a computational perspective, the advantage of restricting our attention to
closed shell states is that their expected energy formulas can be expressed in terms
of positional wave functions alone.

The formula will make use of the function

$$
c : S_n \rightarrow \mathbb{Z}
$$

which maps a permutation $\sigma\in S_n$ to the number of cycles in the
[cyclic decomposition](https://en.wikipedia.org/wiki/Permutation#Cycle_notation)
of $\sigma$.

> **Claim (Closed Shell Expectation).**
> Let $n\in\mathbb{Z}$ be a positive integer, 
> $T^k\in\mathrm{End}(L^2(\mathbb{R}^3)^{\otimes k})$
> a symmetric operator and $T^{2n}\in\mathrm{End}(\Lambda^{2n}\mathcal{H})$ the symmetric extension
> of $T^k\otimes\mathrm{Id}^{\otimes k} \in \mathrm{End}(\mathcal{H}^{\otimes k})$.
>
> Let $\|\psi_1\rangle,\dots,\|\psi_n\rangle\in L^2(\mathbb{R}^3)$ 
> be orthonormal positional wave functions.
>
> Then:
>
> $$
> \langle \psi_1,\dots,\psi_n | T^{2n} | \psi_1,\dots,\psi_n \rangle =
> \frac{1}{k!}
> \sum_{I \in [n]^k}\sum_{\sigma\in S_k}
> 2^{c(\sigma)}\mathrm{sgn}(\sigma)\langle \psi_I | T^k P_{\sigma}| \psi_I \rangle 
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

For each bit-string $s\in [2]^n$, we'll define the spin state:

$$
|s\rangle := |s_1\dots s_n\rangle \in \left(\mathbb{C}^2\right)^{\otimes n}
$$

By XXX and the definition of $T^{2n}$:

$$
\langle \psi_1,\dots,\psi_n | T^{2n} | \psi_1,\dots,\psi_n \rangle =
\frac{1}{k!} 
\sum_{I\in [n]^k} 
\sum_{s \in [2]^k}
\sum_{\sigma\in S_n}
\langle \psi_I | T^k P_\sigma | \psi_I \rangle 
\langle s_I | P_\sigma | s_I \rangle 
$$

Let $t\in [2]^k$ be a bit-string and $\sigma\in S_k$ a permutation.

By the orthonormality of $\|0\rangle,\|1\rangle\in\mathbb{C}^2$:

$$
\langle t | P_\sigma | t \rangle =
\begin{cases}
1 & t = \sigma(t) \\
0 & \mathrm{else}
\end{cases}
$$

If $\sigma$ is a cycle, then $t = \sigma(t)$ if and only if $t$ is all zeros or all ones.
I.e, $t=(0,\dots,0)$ or $t=(1,\dots,1)$. This implies that for a general permutation 
$\sigma\in S_k$, the number of bit-strings $t\in [2]^k$ such that $t = \sigma(t)$
is equal to $2^{c(\sigma)}$.

Therefore:

$$
\sum_{s \in [2]^k}
\langle s_I | P_\sigma | s_I \rangle = 2^{c(\sigma)}
$$

_q.e.d_

</div>
</details>

The following corollary is useful for computing the norm of a closed shell Slater determinant.

> **Claim (Closed Shell Norm).**
> Let $n\in\mathbb{Z}$ be a positive integer and
> $\|\psi_1\rangle,\dots,\psi_n\rangle\in L^2(\mathbb{R}^3)$ 
> orthonormal single-electron positional wave functions.
>
> Then:
>
> $$
> \langle \psi_1,\dots,\psi_n | \psi_1,\dots,\psi_n \rangle = 
> \frac{1}{n}\sum_{i=1}^n \langle \psi_i | \psi_i \rangle
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

We'll apply XXX to the single-electron identity operator
$T^1 = \mathrm{Id}\in\mathrm{End}(L^2(\mathbb{R}^3))$.

Let $T^{2n}\in\mathrm{End}(\Lambda^{2n}\mathcal{H})$ be the symmetric extension of
$T^1\otimes\mathrm{Id}\in\mathrm{End}(\mathcal{H})$.

By the definition symmetric extension:

$$
\begin{align*}
T^{2n} &= \frac{1}{(2n - 1)!}\sum_{\sigma\in S_{2n}}P_{\sigma^{-1}}T_1^1 P_\sigma \\
&= \frac{1}{(2n - 1)!}\sum_{\sigma\in S_{2n}}\mathrm{Id} \\
&= 2n\mathrm{Id}
\end{align*}
$$

Therefore:

$$
\langle \psi_1,\dots,\psi_n | T^{2n} | \psi_1,\dots,\psi_n \rangle = 
2n \langle \psi_1,\dots,\psi_n | \psi_1,\dots,\psi_n \rangle 
$$

On the other hand, by claim XXX:

$$
\begin{align*}
\langle \psi_1,\dots,\psi_n | T^{2n} | \psi_1,\dots,\psi_n \rangle 
&= \sum_{i=1}^n 2 \langle \psi_i | T^1 | \psi_i \rangle \\
&= 2 \sum_{i=1}^n \langle \psi_i | \psi_i \rangle
\end{align*}
$$

The claim follows by combining XXX and XXX.

_q.e.d_
</div>
</details>

For the remainder of this post, we'll work exclusively with closed shell states.

## The Electronic Hamiltonian

The goal of this section is to construct an operator that represents energy of 
the electrons in a molecule.

Since electrons have negative charge and nuclei have positive charge, the electronic energy 
can be expressed as a sum of the kinetic energies of the electrons,
the potential energies of the electron-electron repulsions,
and the potential energies of the electron-nuclei attractions.

We'll start this section by defining self-adjoint operators on 
$\Lambda^n\mathcal{H}$
corresponding to each type of energy.

The electronic
[Hamiltonian](https://en.wikipedia.org/wiki/Hamiltonian_(quantum_mechanics))
will then be defined to be their sum.

### Kinetic Energy

We'll start by defining the kinetic energy operator $T^1$ on $L^2(\mathbb{R}^3)$,
the space of single-electron positional wave functions. Let $\|\psi\rangle\in L^2(\mathbb{R}^3)$
be a positional wave function. By definition:

$$
T^1 |\psi\rangle := -\frac{1}{2}|\nabla^2 \psi \rangle \in L^2(\mathbb{R}^3)
$$

where $\nabla^2$ denotes the
[Laplace operator](https://en.wikipedia.org/wiki/Laplace_operator):

$$
\nabla^2\psi =
\frac{\partial^2}{\partial x^2}\Psi +
\frac{\partial^2}{\partial y^2}\Psi +
\frac{\partial^2}{\partial z^2}\Psi
$$

The Laplace operator
[is self-adjoint](https://en.wikipedia.org/wiki/Self-adjoint_operator#Boundary_conditions)
under the assumption that $\phi(\mathbf{r})$ goes to $0$ as $||\mathbf{r}||$ goes to infinity.

We can extend $T^1\in \mathrm{End}(L^2(\mathbb{R}^3))$ to an operator
$T^1\otimes\mathrm{Id}\in\mathrm{End}(\mathcal{H})$ that acts trivially on the spin component.

The kinetic energy of $n$-electrons, denoted by $T^n\in\mathrm{End}(\Lambda^n\mathcal{H})$,
is defined to be the [symmetric extension](XXX) of
$T^1\otimes\mathrm{Id}$ from $\mathrm{End}(\mathcal{H})$ to $\mathrm{End}(\Lambda^n\mathcal{H})$.

The expected kinetic energy of a closed shell Slater determinant follows immediately from XXX.

> **Claim (Kinetic Energy Expectation).**
> Let $n\in\mathbb{Z}$ be a positive integer and
> $\|\psi_1\rangle,\dots,\|\psi_n\rangle\in L^2(\mathbb{R}^3)$
> be orthonormal single-electron positional wave functions.
>
> The expected kinetic energy of the closed shell Slater determinant
> $\|\psi_1,\dots,\psi_n\rangle\in\Lambda^n \mathcal{H}$ is given by:
>
> $$
> \langle \psi_1,\dots,\psi_n | T^n | \psi_1,\dots,\psi_n \rangle
> = \sum_{i=1}^n \langle \psi_i | T^1 | \psi_i \rangle
> $$

The inner products $\langle \psi_i | T^1 | \psi_i \rangle$ can be evaluated
using the definition of $T^1$ and the definition of the inner product on
$L^2(\mathbb{R}^3)$.

> **Claim (Kinetic Energy Integral).**
> Let $\|\psi_1\rangle,\|\psi_2\rangle\in L^2(\mathbb{R}^3)$ 
> be single-electron wave functions.
>
> Then:
>
> $$
> \langle \psi_1 | T^1 | \psi_2 \rangle = 
> -\frac{1}{2}
> \int_{\mathbb{R}^3} \psi_1^*(\mathbf{r})\nabla^2\psi_2(\mathbf{r})d\mathbf{r}
> $$

### Electron Nuclear Attraction

We'll now consider the potential energy operator corresponding to the 
[Coulomb](https://en.wikipedia.org/wiki/Coulomb%27s_law) attraction
between electrons and nuclei.

Suppose there are $m$ nuclei with positions
$\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_m)\in\mathbb{R}^{3m}$ and
[atomic numbers](https://en.wikipedia.org/wiki/Atomic_number)
$Z = (Z_1,\dots,Z_m)\in\mathbb{Z}^m$.

The nuclei attraction operator for single-electron positional wave functions
is an operator 
$V^1_{\mathrm{en}}(\cdot;\mathbf{R},Z)\in\mathrm{End}(L^2(\mathbb{R}^3))$ that acts as
multiplication by

$$
\phi_{\mathrm{en}}(\mathbf{r}; \mathbf{R},Z) := -\sum_{i=1}^m \frac{Z_i}{||\mathbf{R}_i - \mathbf{r}||}
\in L^2(\mathbb{R}^3)
$$

Specifically, given a positional wave function $\psi(\mathbf{r})\in L^2(\mathbb{R}^3)$:

$$
V^1_{\mathrm{en}}(\cdot; \mathbf{R},Z)|\psi(\mathbf{r})\rangle := 
|\phi_{\mathrm{en}}(\mathbf{r}; \mathbf{R},Z)\cdot\psi(\mathbf{r})\rangle
\in L^2(\mathbb{R}^3)
$$

Since $V^1\_{\mathrm{en}}(\cdot; \mathbf{R},Z)$ acts on the $L^2(\mathbb{R}^3)$ by multiplication
by the constant $\phi_{\mathrm{en}}$, it is self-adjoint.

As usual, we can extend
$V^1\_{\mathrm{en}}(\cdot; \mathbf{R},Z)\in\mathrm{End}(L^2(\mathbb{R}^3))$ to an operator
$V^1\_{\mathrm{en}}(\cdot; \mathbf{R},Z)\otimes\mathrm{Id}\in\mathrm{End}(\mathcal{H})$ that acts
trivially on the spin component.

The operator corresponding to the attraction between $n$ electrons and the $m$
nuclei, denoted by $V_{\mathrm{en}}^n(\cdot; \mathbf{R},Z)\in\mathrm{End}(\Lambda^n\mathcal{H})$ 
is defined to be the symmetric extension of $V_{\mathrm{en}}^1(\cdot; \mathbf{R},Z)\otimes\mathrm{Id}$ from
$\mathrm{End}(\mathcal{H})$ to $\mathrm{End}(\Lambda^n\mathcal{H})$.

The nuclear attraction energy of a closed shell Slater determinant also follows
immediately from XXX.

> **Claim (Nuclear Attraction Expectation).**
> Let $m,n\in\mathbb{Z}$ be positive integers.
>
> Let $\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_m)\in\mathbb{R}^{3m}$ 
> be $m$ nuclear positions and $Z=(Z_1,\dots,Z_m)\in\mathbb{Z}^m$ be $m$
> nuclear numbers.
>
> Let $\|\psi_1\rangle,\dots,\|\psi_n\rangle\in L^2(\mathbb{R}^3)$
> be orthonormal single-electron positional wave functions.
>
> Then the expected nuclear attraction energy of the closed shell Slater determinant
> $\|\psi_1,\dots,\psi_n\rangle\in\Lambda^n \mathcal{H}$ is given by:
>
> $$
> \langle \psi_1,\dots,\psi_n | V^n_{\mathrm{en}}(\cdot; \mathbf{R},Z) | \psi_1,\dots,\psi_n \rangle
> = \sum_{i=1}^n \langle \psi_i | V^1_{\mathrm{en}}(\cdot; \mathbf{R},Z) | \psi_i \rangle
> $$

The single-electron inner-products can be evaluated using the definition of the inner product
on $L^2(\mathbb{R}^3)$:

> **Claim (Nuclear Attraction Integral).**
> Let $m,n\in\mathbb{Z}$ be positive integers.
>
> Let $\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_m)\in\mathbb{R}^{3m}$ 
> be $m$ nuclear positions and $Z=(Z_1,\dots,Z_m)\in\mathbb{Z}^m$ be $m$
> nuclear numbers.
>
> Let $\|\psi_1\rangle,\|\psi_2\rangle\in L^2(\mathbb{R}^3)$
> be single-electron wave functions.
>
> Then:
>
> $$ 
> \langle \psi_1 | V_{\mathrm{en}}^1(\cdot; \mathbf{R},Z) | \psi_2 \rangle =
> -\sum_{j=1}^m \int_{\mathbb{R}^3} 
> \psi_1^*(\mathbf{r}) \frac{Z_j}{||\mathbf{R}_j - \mathbf{r}||}\psi_2(\mathbf{r})
> d\mathbf{r}
> $$

### Electron Repulsion

We'll now consider the potential energy corresponding to the Coulomb repulsion
between pairs of electrons.

First we'll define Coulomb repulsion potential for a $2$-electron positional wave function
as a symmetric operator $V_{\mathrm{ee}}^2\in\mathrm{End}(L^2(\mathbb{R}^3)^{\otimes 2}$.

For this purpose,
we'll use the canonical isomorphism between $L^2(\mathbb{R}^3) \otimes L^2(\mathbb{R}^3)$
and $L^2(\mathbb{R}^3\times\mathbb{R}^3)$:

$$
\begin{align*}
F: L^2(\mathbb{R}^3)\otimes L^2(\mathbb{R}^3) &\rightarrow L^2(\mathbb{R}^3\times\mathbb{R}^3) \\
(f(\mathbf{r}), g(\mathbf{r})) &\mapsto f(\mathbf{r}_1)\cdot g(\mathbf{r}_2)
\end{align*}
$$

See
[wikipedia](https://en.wikipedia.org/wiki/Tensor_product_of_Hilbert_spaces#Examples_and_applications)
for more information about the canonical isomorphism.

The operator $V_{\mathrm{ee}}^2$ is defined on 
$L^2(\mathbb{R}^3\times\mathbb{R}^3)$ by multiplication by
$\frac{1}{||\mathbf{r}_1 - \mathbf{r}_2||}$:

$$
V_{\mathrm{ee}}^2(\psi(\mathbf{r}_1,\mathbf{r}_2)) :=
\frac{1}{||\mathbf{r}_1 - \mathbf{r}_2||} \cdot \psi(\mathbf{r}_1,\mathbf{r}_2)
$$

Clearly $V_{\mathrm{ee}}^2$ commutes with the transposition of the coordinates
$\mathbf{r}\_1$ and $\mathbf{r}\_2$. Therefore, under the canonical isomorphism
$V_{\mathrm{ee}}^2$ is a symmetric operator on
$L^2(\mathbb{R}^3)^{\otimes 2}$.

We can extend $V_{\mathrm{ee}}^2$ to a symmetric operator on 
$V_{\mathrm{ee}}^2\otimes\mathrm{Id}^{\otimes 2}\in\mathrm{End}(\mathcal{H}^{\otimes 2})$
that acts trivially on the spin factors $(\mathbb{C}^2)^{\otimes 2}$. 

The Coulomb repulsion operator for $n$-electrons, denoted
$V_{\mathrm{ee}}^n\in\mathrm{End}(\Lambda^n\mathcal{H})$, is defined as the
symmetric extension of $V_{\mathrm{ee}}^2\otimes\mathrm{Id}^{\otimes 2}$ from
$\mathrm{End}(\Lambda^2\mathcal{H})$ to $\mathrm{End}(\Lambda^n\mathcal{H})$.

We can again apply XXX to compute the expected election repulsion energy of
a closed shell Slater determinant.

> **Claim (Electron Repulsion Expectation).**
> Let $n\in\mathbb{Z}$ be a positive integer.
> Let $\|\psi_1\rangle,\dots,\|\psi_n\rangle\in L^2(\mathbb{R}^3)$
> be orthonormal  single-electron positional wave functions.
>
> The expected electron repulsion energy of the closed shell Slater determinant 
> $\|\psi_1,\dots,\psi_n\rangle\in\Lambda^n \mathcal{H}$ is given by:
>
> $$
> \langle \psi_1,\dots,\psi_n | V_{\mathrm{ee}}^n | \psi_1,\dots,\psi_n \rangle
> = \sum_{i,j=1}^n \left(
> \langle \psi_i\psi_j | V_\mathrm{ee}^2 | \psi_i\psi_j \rangle -
> \frac{1}{2}\langle \psi_i\psi_j | V_\mathrm{ee}^2 | \psi_j\psi_i \rangle \right)
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The claim follows immediately from claim XXX:

$$
\begin{align*}
\langle \psi_1,\dots,\psi_n | V_{\mathrm{ee}}^n | \psi_1,\dots,\psi_n \rangle
&= \frac{1}{2}\sum_{i,j=1}^n \sum_{\sigma\in S_2}
2^{c(\sigma)}\mathrm{sgn}(\sigma)
\langle \psi_i\psi_j |V_\mathrm{ee}^2 P_\sigma | \psi_i\psi_j \rangle \\
&= \sum_{i,j=1}^n\left(
\langle \psi_i\psi_j | V_\mathrm{ee}^2 | \psi_i\psi_j \rangle -
\frac{1}{2}\langle \psi_i\psi_j | V_\mathrm{ee}^2 | \psi_j\psi_i \rangle\right)
\end{align*}
$$

_q.e.d_

</div>
</details>

The two-electron inner-products in the above claim can be computed as follows.

> **Claim (Electron Repulsion Integral).**
> Let $n\in\mathbb{Z}$ be a positive integer.
>
> Let $\|\psi_1\rangle,\dots,\|\psi_4\rangle\in L^2(\mathbb{R}^3)$
> be single-electron wave functions.
>
> Then:
>
> $$
> \langle \psi_1\psi_2 | V_\mathrm{ee}^2 | \psi_3\psi_4 \rangle =
> \int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
> \psi_1^*(\mathbf{r}_1)\psi_2^*(\mathbf{r}_2)
> \frac{1}{||\mathbf{r}_1 - \mathbf{r}_2||}
> \psi_3(\mathbf{r}_1)\psi_4(\mathbf{r}_2)
> d\mathbf{r}_1 d\mathbf{r}_2
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The claim follows from the definition of $V_\mathrm{ee}^2$ and the definition
of the inner-product on $L^2(\mathbb{R}^3\times\mathbb{R}^3)$.

_q.e.d_

</div>
</details>

### The Hamiltonian

Consider a molecule with $m$ nuclei and $n$ electrons.
Suppose that the nuclei have atomic numbers
$Z = (Z_1,\dots,Z_m)\in\mathbb{Z}^m$ and positions
$\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_m)\in\mathbb{R}^{3m}$.

The electronic Hamiltonian of the molecule,
denoted $H^n(\cdot;\mathbf{R},Z) \in \mathrm{End}(\Lambda^n\mathcal{H})$,
is defined to be the sum of the operators defined above:

$$
H^n(\cdot;\mathbf{R},Z) := 
T^n + V_{\mathrm{en}}^n(\cdot;\mathbf{R},Z) + V_{\mathrm{ee}}^n
$$

## The Schrodinger Equation

The 
[time-independent Schrodinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation#Time-independent_equation)
determines the
[stationary states](https://en.wikipedia.org/wiki/Stationary_state)
of a system with Hilbert space $V$ and 
[Hamiltonian](https://en.wikipedia.org/wiki/Hamiltonian_(quantum_mechanics))
$H\in\mathrm{End}(V)$.

Specifically, it says that if $|\Psi\rangle\in V$
is a stationary state with electronic energy 
$E\in\mathbb{R}$, then $|\Psi\rangle$ is an eigenvector of $H$
with eigenvalue $E$:

$$
H|\Psi\rangle = E|\Psi\rangle
$$

In particular, the ground state of the system is given by the eigenvector of $H$
with the smallest eigenvalue, denoted $E_0$.

In theory, all we need to do to determine the electronic ground state of a molecule
is to diagonalize the Hamiltonian
$H^n(\cdot;\mathbf{R},Z)\in\mathrm{End}(\Lambda^n\mathcal{H})$
and find the eigenvector with the smallest eigenvalue.
In practice this is infeasible for all but the simplest systems.

To see why, note that the state space of $n$-electrons is equal to

$$
\Lambda^n\mathcal{H} = \Lambda^n(L^2(\mathbb{R}^3)\otimes\mathbb{C}^2)
$$

The set of integrable functions $L^2(\mathbb{R}^3)$ 
is infinite which means that we cannot directly express
$H^n(\cdot;\mathbf{R},Z)\in\mathrm{End}(\Lambda^n\mathcal{H})$ as a matrix. 

One idea could be to discretize $\mathbb{R}$ into a finite set of
points and express an orbital $\|\Psi\rangle$ in terms of the vector of its values on each point.
However, even with a conservative discretization of only $100$ points per dimension,
$L^2(\mathbb{R}^3)\otimes\mathbb{C}^2$ is $2 \cdot 100^3$ dimensional and so
$\Lambda^n(L^2(\mathbb{R}^3)\otimes(\mathbb{C}^2)$ is $\binom{2 \cdot 100^3}{n}$
dimensional which is still quite large.

Rather than finding the exact ground state, we'll instead use the
[Variational Principle](https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics))
to approximate it.

## The Variational Principle

Consider a quantum system with Hilbert space $V$ and Hamiltonian $H\in\mathrm{End}(V)$.
The [Variational Principle](https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics))
states that for any state $\|\Psi\rangle\in V$ satisfying $\langle\Psi \| \Psi \rangle = 1$,
the expectation energy of $\|\Psi\rangle$ is an upper bound on the
ground state energy $E_0$:

$$
\langle \Psi | H | \Psi \rangle \geq E_0
$$

Furthermore, the inequality becomes exact when $\|\Psi\rangle$ is the ground state.

In our context, consider a molecule that has $m$ nuclei with atomic
numbers $Z=(Z_1,\dots,Z_m)$ and positions $\mathbf{R}=(\mathbf{R}_1,\dots,\mathbf{R}_m)$,
together with $n$ electrons. Let
$H = H^n(\mathbf{R}, Z)\in\mathrm{End}(\Lambda^n\mathcal{H})$
be the [electronic Hamiltonian](XXX) of the molecule.

According to the variational principle, in order to find the electronic ground state
we can search for an $n$-electron state
$\|\Psi\rangle\in\Lambda^n\mathcal{H}$ with unit norm that minimizes the expectation energy

$$
\langle \Psi | H^n(\mathbf{R}, Z) | \Psi \rangle
$$

There are two major obstacles to this approach.

First, as mentioned in section [The Schrodinger Equation](#schrodinger-equation),
the space of $n$-electron states, $\Lambda^n\mathcal{H}$, is infinite dimensional.

The second issue is that, given a state $\|\Psi\rangle\in\Lambda^n\mathcal{H}$,
evaluating the inner product
$\langle \Psi | H^n(\mathbf{R},Z) | \Psi \rangle$
involves computing an integral over $\mathbb{R}^{3\times n}$ that is basically impossible to
solve numerically.

As a first step to making the problem more tractable, we'll restrict our search
to $n$-electron states that are equal to the closed shell Slater determinant of 
$n/2$ orthonormal single-electron positional wave functions.
In other words, we'll assume that
$\|\Psi\rangle$ is of the form:

$$
|\Psi\rangle = |\psi_1,\dots,\psi_{n/2}\rangle \in \Lambda^n\mathcal{H}
$$

where $\|\psi_1\rangle,\dots,\|\psi_{n/2}\rangle\in L^2(\mathbb{R}^3)$
are orthonormal.

Let $E : L^2(\mathbb{R}^3)^{n/2} \rightarrow \mathbb{R}$ 
denote the expectation energy of the Slater determinant $|\Psi\rangle$
as a function of the single-electron wave functions:

$$
E(|\psi_1\rangle,\dots,|\psi_{n/2}\rangle) := 
\langle\psi_1,\dots,\psi_{n/2} | H | \psi_1,\dots,\psi_{n/2}\rangle
$$

We can restate our search as a constrained minimization of $E$. Specifically,
our objective is to find single-electron wave functions
$\|\psi_1\rangle,\dots,\|\psi_{n/2}\rangle\in L^2(\mathbb{R}^3)$ that minimize
$E(|\psi_1\rangle,\dots,|\psi_{n/2}\rangle)$ subject to the orthonormality constraint:

$$
\langle \psi_i | \psi_j \rangle = \delta_{ij}
$$

The point of this reformulation is that we can solve this optimization problem
using the method of
[Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier)

First, we'll introduce $(n/2)^2$ Lagrange multipliers $\lambda_{kl}\in\mathbb{R}$
where $1 \leq k,l \leq n/2$. We'll then introduce the Lagrangian:

$$
L: L^2(\mathbb{R}^3)^{n/2} \times \mathbb{R}^{n/2 \times n/2}
$$

defined by:

$$
L(|\psi_1\rangle,\dots,|\psi_{n/2}\rangle,\lambda_{0,0},\dots,\lambda_{n/2,n/2}) :=
E(|\psi_1\rangle,\dots,|\psi_{n/2}\rangle) - 
\sum_{k,l=0}^{n/2}\lambda_{kl}(\langle \psi_k | \psi_l \rangle - \delta_{kl})
$$

By the theory of Lagrange multipliers, the minima of $E$ subject to the orthonormality
constraints are critical points of $L$. In particular, if
$\|\psi_1\rangle,\dots,\|\psi_{n/2}\rangle\in L^2(\mathbb{R}^3)$
is a solution to the optimization problem, then there exist Lagrange multipliers
$\lambda_{kl}\in\mathbb{R}$ such that for all $1 \leq i \leq n/2$:

$$
\delta
L(|\psi_1\rangle,\dots,|\psi_{n/2}\rangle,\lambda_{0,0},\dots,\lambda_{n/2,n/2}) = 0
$$

By linearity of the first variation, this is equivalent to:

$$
\delta E = \sum_{k,l=0}^{n/2}\lambda_{kl} \delta \langle \psi_k | \psi_l \rangle \right)
$$

We can use the results of section XXX to compute the left hand side in terms of
single and double electron operators. To facilitate notation, we'll denote the sum
of the single-electron kinetic energy and electron-nuclear attraction operators
from section XXX by $H^1(\mathbb{R}, Z)$:

$$
H^1(\mathbb{R}, Z) := 
T^1 + V^1_{\mathrm{en}}(\mathbb{R}, Z) \in \mathrm{End}(L^2(\mathbb{R}^3))
$$

> **Definition (First Variation).**
> Let $n\in\mathbb{Z}$ be a positive integer, let $V_1,\dots,V_n$ be Hilbert spaces
> and $f:\Prod_i=1^nV_i \rightarrow \mathbf{C}$ a function.
>
> The _first variation_ of $f$ at $(v_1,\dots,v_n)\in\Prod_i=1^nV_i$ is a linear function
> $\delta f(v_1,\dots,v_n)\in\mathrm{Hom}(\Prod_i=1^nV_i,\mathbb{C}$
> that satisfies the property that for all $(\delta v_1,\dots,\delta v_n)\in\Prod_i=1^nV_i$:
>
> $$
> f(v_1+\delta v_1,\dots,v_n+\delta v_n) = 
> f(v_1,\dots,v_n) + \delta f(v_1,\dots,v_n)[\delta v_1,\dots,\delta v_n]
> + O(\sum_{i=1}^n||\delta v_i||^2)
> $$

> **Claim (Expectation First Variation).**
> Let $V$ be a Hilbert space, $A\in\mathrm{End}(V)$ a self-adjoint operator on $V$ and
> $f:V\rightarrow\mathbb{C}$ the function defined by:
>
> $$
> f(v) := \langle v | A | v \rangle
> $$
>
> Then
>
> $$
> \delta f(v)[\delta v] = 2\mathrm{Re}(\langle \delta v | A | v \rangle)
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

Consider the function $B: V \times V \rightarrow \mathbb{C}$
defined by:

$$
B(v_1,v_2) := \langle v_1 | A | v_2 \rangle
$$

The first variation of $B$ is given by:
$$
\delta B = \langle\delta v_1 | A | v_2 \rangle + \langle v_1 | A | \delta v_2 \rangle
$$

Let $g:V \rightarrow V\times V$ denote the diagonal
map defined by $g(v) = (v,v)$. Applying the chain rule
to $B\circ g$:

$$
\begin{align*}
\delta\langle v | A | v \rangle =
&= \langle \delta v | A | v \rangle + \langle v | A | \delta v \rangle \\
&= \langle \delta v | A | v \rangle + \langle \delta v | A | v \rangle^* \\
&= 2\mathrm{Re}(\langle \delta v | A | v \rangle)
\end{align*}
$$

_q.e.d_
</div>
</details>

> **Claim (Double Expectation First Variation).**
> Let $V$ be a Hilbert space and $A\in\mathrm{End}(V^{\otimes 2})$ an operator on $V^{\otimes 2}$.
> Let $f:V^{\otimes 2}\rightarrow\mathbb{C}$ be defined by:
>
> $$
> f(v,w) := \langle v w | A | v w \rangle
> $$
>
> Then
>
> $$
> \delta f(v,w)[\delta v,\delta w] = 2\mathrm{Re}(
>    \langle \delta v w | A | v w \rangle + \langle \delta w v | A | w v \rangle
> )
> $$

When the arguments of $\delta f$ are clear from the context, we will omit them.
For example, we'll sometimes write the statement of claim XXX as:

$$
\delta \langle v | A | v \rangle = 2\mathrm{Re}(\langle \delta v | A | v \rangle)
$$

> **Definition (Coulomb Operator).**
> Let $\rho\in\mathrm{End}(L^2(\mathbb{R}^3))$ be a density operator.
>
> The _Coulomb operator_ associated to $\rho$ is an operator on 
> $L^2(\mathbb{R}^3)$ defined by:
>
> $$
> J(\rho) :=
> \mathrm{Tr}_2 \left(
> (\mathrm{Id}^1 \otimes \rho) V_\mathrm{ee}^2 \right)
> \in\mathrm{End}(L^2(\mathbb{R}^3))
> $$

> **Definition (Exchange Operator).**
> Let $\rho\in\mathrm{End}(L^2(\mathbb{R}^3))$ be a density operator.
>
> The _exchange operator_ associated to $\rho$ is an operator on 
> $L^2(\mathbb{R}^3)$ defined by:
>
> $$
> K(\rho) :=
> \mathrm{Tr}_2 \left(
> (\mathrm{Id}^1 \otimes \rho) V_\mathrm{ee}^2 P_{(1,2)} \right)
> \in\mathrm{End}(L^2(\mathbb{R}^3))
> $$

> **Claim.**
> Let $m\in\mathbb{Z}$ be a positive integer,
> $Z=(Z_1,\dots,Z_m)\in\mathbb{Z}^m$ atomic numbers and
> $\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_m)\in\mathbb{R}^{3\times m}$
> nuclear positions.
>
> Let $n\in\mathbb{Z}$ be an even integer and let
> $\|\psi_1\rangle,\dots,|\psi_{n/2}\rangle\in L^2(\mathbb{R}^3)$
> be single-electron positional states.
>
> Let $\rho\in\mathrm{End}(L^2(\mathbb{R}^3))$ denote the corresponding 
> electron density operator:
>
> $$
> \rho := 2\sum_{i=1}^{n/2}|\psi_i\rangle\langle\psi_i| 
> \in\mathrm{End}(L^2(\mathbb{R}^3))
> $$
>
> Then, for all $1\leq i \leq n/2$$:
>
> $$
> \delta E =
> \mathrm{Re}\left( 
>   \sum_{i=1}^{n/2}\langle \delta\psi_i | H^1(\mathbf{R}, Z) + 2J(\rho) - K(\rho) | \psi_i\rangle
> \right)
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

By definition,

$$
H^n = T^n +  V_\mathrm{en}^n(\mathbf{R}, Z) + V_\mathrm{ee}^n
$$

Recall from section XXX that $T^n$ and $V_\mathrm{en}^n(\mathbf{R}, Z)$
are defined to be the symmetric extensions of the single-electron operators
$T^1\in\mathrm{End}(L^2(\mathbb{R}^3))$ and 
$V_\mathrm{en}^1\in\mathrm{End}(L^2(\mathbb{R}^3))$.

Similarly, $V_\mathrm{ee}^n$ is defined to be the symmetric extension of the
double electron operator
$V_\mathrm{ee}^2\in\mathrm{End}(L^2(\mathbb{R}^3)^{\otimes 2})$.

We'll split the functional $E$ into two components by defining

$$
E_1(\psi_1,\dots,\psi_{n/2}) := 
\langle \psi_1,\dots,\psi_{n/2} | T^n  + V_\mathrm{en}^n(\mathbf{R}, Z) |
\psi_1,\dots,\psi_{n/2} \rangle
$$

and

$$
E_2(\psi_1,\dots,\psi_{n/2}) := 
\langle \psi_1,\dots,\psi_{n/2} | V_\mathrm{ee}^n |
\psi_1,\dots,\psi_{n/2} \rangle
$$

We'll start by evaluating $\delta_i E_1$.

By XXX:

$$
E^1(\psi_1,\dots,\psi_{n/2}) =
\sum_{j=1}^{n/2}\langle \psi_j | H^1 \psi_j \rangle
$$

Therefore, by XXX:

$$
\delta E^1 &= 2\mathrm{Re}\left(
   \sum_{j=1}^{n/2}\langle \delta\psi_i | H^1 | \psi_i \rangle
\right)
$$

The analysis for $E^2$ is similar. To facilitate notation, we'll define:

$$
H^2 := V_\mathrm{ee}^2(\mathrm{Id} - \frac{1}{2}P_{(1,2)})
$$

By XXX:

$$
E_2(\psi_1,\dots,\psi_{n/2}) = 
\sum_{i,j=1}^{n/2}(
    \langle\psi_i\psi_j | H^2 | \psi_i\psi_j\rangle -
)
$$

Therefore, by XXX:

$$
\delta E_2 = 
2\mathrm{Re}\left(
\sum_{i,j=1}^{n/2}(
    2\langle\delta\psi_i\psi_j | H^2 | \psi_i\psi_j\rangle -
)\left)
$$

Finally, note that for all $1\leq i \leq n/2$:

$$
\begin{align*}
\sum_{j=1}^{n/2}(
    \langle\delta\psi_i\psi_j | H^2 | \psi_i\psi_j\rangle
&= \langle \delta\psi_i | \sum_{j=1}^{n/2}(
    (\mathrm{Id}\otimes\langle\psi_j |) H^2 (\mathrm{Id}\otimes |psi_j\rangle) |\psi_i\rangle \\
&= \langle \delta\psi_i | \mathrm{Tr}_2 (\rho H^2) | \psi_i \rangle \\
&= \langle \delta\psi_i | J(\rho) - \frac{1}{2}K(\rho) | \psi_i \rangle
\end{align*}

_q.e.d_
</div>
</details>


> **Definition (Fock Operator).**
> Let $m\in\mathbb{Z}$ be a positive integer,
> $Z=(Z_1,\dots,Z_m)\in\mathbb{Z}^m$ atomic number and
> $\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_m)\in\mathbb{R}^{3\times m}$
> nuclear positions.
>
> Let $\rho\in\mathrm{End}(L^2(\mathbb{R}^3))$ be a density operator.
>
> The _Fock operator_ associated to the molecule $(\mathbf{R}, Z)$ and electron
> density $\rho$ is an operator on $L^2(\mathbb{R}^3)$:
>
> $$
> F(\rho, \mathbf{R}, Z) := H^1(\mathbf{R}, Z) + J(\rho) - \frac{1}{2}K(\rho)
> \in\mathrm{End}(L^2(\mathbb{R}^3))
> $$

## Linear Variation

Consider a molecule with $m$ nuclei and $2n$ electrons for some positive
integers $m,n\in\mathbb{Z}$. Denote the atomic numbers of the nuclei by
$Z = (Z_1,\dots,Z_m)\in\mathbb{Z}^m$.

In addition, let
$B\subset L^2(\mathbb{R}^3)$ be a finite subset of linearly independent
positional wave functions of size $b := |B|$.

As discussed in the previous section, our strategy for approximating the ground state
of the molecule is to search for nuclei positions

$$
\mathbf{R} = (\mathbf{R}_1,\dots,\mathbf{R}_m)\in\mathbb{R}^{3\times m}
$$

and positional wave functions

$$
|\psi_1\rangle,\dots,|\psi_n\rangle\in\mathrm{Span}(B)\subset L^2(\mathbb{R}^3)
$$

which minimize the expected energy of the closed shell Slater determinant

$$
|\Psi\rangle := |\psi_1,\dots,\psi_n\rangle \in \Lambda^{2n}(\mathcal{H})
$$

subject to the constraint that the wave functions
$\|\psi_1\rangle,\dots,\|\psi_n\rangle$ are
orthonormal.

By XXX, the expected energy of the molecule when the nuclei are in positions
$\mathbf{R}\in\mathbb{R}^{3\times m}$ and the electrons are in state
$\|\Psi\rangle$ is equal to:

$$
F(\Psi, \mathbf{R}; Z) := 
\frac{\langle \Psi | H(\cdot;\mathbf{R},Z) | \Psi \rangle}{\langle \Psi | \Psi \rangle}
+ V_\mathrm{nuc}(\mathbf{R}, Z)
$$

If $\|\psi_1\rangle,\dots,\|\psi_n\rangle$ are orthonormal, then by XXX:

$$
|\Psi | \Psi \rangle = 1
$$

Therefore, under the orthonormality constraint:

$$
F(\Psi, \mathbf{R}; Z) = 
\langle \Psi | H(\cdot;\mathbf{R},Z) | \Psi \rangle}
+ V_\mathrm{nuc}(\mathbf{R}, Z)
$$

We'll apply the method of
[Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier)
to minimize $F$ under the constraint that 
$\|\psi_1\rangle,\dots,\|\psi_n\rangle$ are orthonormal.

Specifically, 

In this section we'll formulate this optimization problem more in terms of the 
linear coefficients of the wave functions $\|\psi_i\rangle$ in terms of the basis $B$.

We'll denote the elements of $B$ by:

$$
B = \{ |\phi_1\rangle,\dots,|\phi_b\rangle \}
$$

Let $\mathbf{C}\in\mathbb{R}^{n\times b}$ be a $n\times b$ matrix and, for all
$1 \leq i \leq n$ define:

$$
|\psi_i\rangle := \sum_{j=1}^b C_{ij}|\phi_j\rangle
$$

By XXX, the expected energy of the molecule when the nuclei are in positions
$\mathbf{R}\in\mathbb{R}^{3\times m}$ and the electrons are in state
$\|\Psi\rangle := \|\psi_1,\dots,\psi_n\rangle\in\Lambda^{2n}\mathcal{H}$ is equal to:

$$
F(\mathbf{C}, \mathbf{R}; Z) := 
\frac{\langle \Psi | H(\cdot;\mathbf{R},Z) | \Psi \rangle}{\langle \Psi | \Psi \rangle}
+ V_\mathrm{nuc}(\mathbf{R}, Z)
$$

Our goal is to minimize $F$ as a function of $\mathbf{C}$ and $\mathbf{R}$
under the constraint that the wave functions $\|\psi_1\rangle,\dots,\|\psi_n\rangle$ are
orthonormal.

First we'll rewrite $F$ explicitly in terms of the linear coefficients $\mathbf{C}$
and the basis states $|\phi_1\rangle,\dots,\|\phi_b\rangle$.

By XXX, assuming that $\|\psi_1\rangle,\dots,\|\psi_n\rangle$ are
orthonormal:

$$
\begin{align*}
\langle \Psi | H(\cdot;\mathbf{R},Z) | \Psi \rangle
&= \sum_{i=1}^n \langle \psi_i | T^1 + V_\mathrm{en}^1(\cdot;\mathbf{R},Z) | \psi_i \rangle \\
&+ \sum_{i,j=1}^n
   \left( 
   2\langle \psi_i\psi_j | V_\mathrm{ee}^2 | \psi_i\psi_j \rangle
   -\langle \psi_i\psi_j | V_\mathrm{ee}^2 | \psi_j\psi_i \rangle
   \right)
\end{align*}
$$

Substituting

$$
|\psi_i\rangle = \sum_{k=1}^b C_{ik}|\phi_b\rangle
$$

into XXX gives us:

$$
\begin{align*}
\langle \Psi | H(\cdot;\mathbf{R},Z) | \Psi \rangle
&= \sum_{i=1}^n\sum_{s,t=1}^b C_{is}C_{it} \langle \phi_s | T^1 + V_\mathrm{en}^1(\cdot;\mathbf{R},Z) | \phi_t \rangle \\
&+ \sum_{i,j=1}^n\sum_{s,t,u,v=1}^b
   (2C_{is}C_{jt}C_{iu}C_{jv}
    -C_{is}C_{jt}C_{ju}C_{iv})
   \langle \phi_s\phi_t | V_\mathrm{ee}^2 | \phi_u\phi_v \rangle
\end{align*}
$$

# Cartesian Gaussians

We will denote the Cartesian coordinates of a point $\mathbf{A}\in\mathbb{R}^3$ by:

$$
\mathbf{A} = (A_x, A_y, A_z)
$$

Similarly, we'll use [multi-index notation](https://en.wikipedia.org/wiki/Multi-index_notation) 
to denote triples of indices with boldface greek letters:

$$
\boldsymbol{\alpha} = (\alpha_x,\alpha_y,\alpha_z)\in \mathbb{N}^3$:
$$

{: #def:cartesian-gaussian }
> **Definition (Cartesian Gaussian).** 
> Let $\boldsymbol{\alpha}\in\mathbb{N}$ be a multi-index,
> $a\in\mathbb{R}$ a positive real number and
> $\mathbf{A}\in\mathbb{R}^3$.
>
> The Cartesian Gaussian is defined by
>
> $$
> G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A}) := (x - A_x)^{\alpha_x}(y - A_y)^{\alpha_y}(z - A_z)^{\alpha_z} e^{-a ||\mathbf{r} - \mathbf{A}||^2}
> $$

An important property of Cartesian Gaussians is that they can be factorized along the Cartesian 
coordinates into one dimensional Cartesian Gaussians. Specifically, define:

{: #def:1d-cartesian-gaussian }
> **Definition (1d Cartesian Gaussian).**
> Let $i\in\mathbb{Z}$ be a non-negative integer, $a\in\mathbb{R}$ a positive
> real number and $A\in\mathbb{R}$.
>
> The one dimensional Cartesian Gaussian is defined by:
>
> $$
> G_i(x, a, A) := (x - A)^ie^{-a(x - A)^2}
> $$

We can now factor a Cartesian Gaussian as:

{: #clm:cartesian-gaussian-factorization }
> **Definition (Cartesian Gaussian Factorization).** 
> Let $\boldsymbol{\alpha}\in\mathbb{N}$ be a multi-index,
> $a\in\mathbb{R}$ a positive real number and
> $\mathbf{A}\in\mathbb{R}^3$.
>
> Then:
>
>  $$
>  G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A}) = 
>  G_{\alpha_x}(x, a, A_x)G_{\alpha_y}(y, a, A_y)G_{\alpha_z}(z, a, A_z)
>  $$

In the special case where the multi-index $\boldsymbol{\alpha}$ is equal to zero,
$G_\mathbf{0}(\mathbf{r}, a, \mathbf{A})$ has spherical symmetry and we'll call it
a _spherical Gaussian_ and drop the subscript.

{: #def:spherical-gaussian }
> **Definition (Spherical Gaussian).** 
> Let $a\in\mathbb{R}$ a positive real number and
> $\mathbf{A}\in\mathbb{R}^3$.
>
> The spherical Gaussian is defined by
>
> $$
> G(\mathbf{r}, a, \mathbf{A}) := e^{-a ||\mathbf{r} - \mathbf{A}||^2}
> $$

The one dimensional spherical Gaussian is defined similarly.

{: #def:1d-spherical-gaussian-integral }
> **Definition (1d Spherical Gaussian).**
> Let $a\in\mathbb{R}$ a positive real number and $A\in\mathbb{R}$.
>
> The one dimensional spherical Gaussian is defined by:
>
> $$
> G(x, a, A) := e^{-a(x - A)^2}
> $$

We'll make heavy use of the classic
[Gaussian integral](https://en.wikipedia.org/wiki/Gaussian_integral):

{: #clm:1d-spherical-gaussian-integral }
> **Claim (1d Spherical Gaussian Integral).**
> Let $a\in\mathbb{R}$ a positive real number and $A\in\mathbb{R}$.
>
> Then:
>
> $$
> \int_{-\infty}^{\infty} G(x, a, A) dx = \sqrt{\frac{\pi}{a}}
> $$

Note that the integral does not depend on $A$.

It follows from
[Cartesian Gaussian Factorization](#clm:cartesian-gaussian-factorization) that:

{: #clm:spherical-gaussian-integral }
> **Claim (Spherical Gaussian Integral).**
> Let $a\in\mathbb{R}$ a positive real number and
> $\mathbf{A}\in\mathbb{R}^3$.
>
> Then:
>
> $$
> \int_\mathbb{R}G(\mathbf{r}, a, \mathbf{A}) = \left(\frac{\pi}{a}\right)^{3/2}
> $$

We'll also record the following derivatives of one-dimensional Cartesian Gaussians
which are easy to compute using the product and chain rules:

{: #clm:1d-spherical-gaussian-derivatives }
> **Claim (1d Cartesian Gaussian Derivatives).**
> Let $i\in\mathbb{Z}$ be a non-negative integer,
> $a\in\mathbb{R}$ a positive real number and $A\in\mathbb{R}$.
>
> Define:
>
> $$
> G_i := G_i(x, a, A)
> $$
>
> Then:
>
> $$
> \begin{align*}
> \frac{d}{dx} G_i &= iG_{i-1} - 2aG_{i+1} \\
> \frac{d^2}{dx^2} G_i &= i(i-1)G_{i-2} - 2a(2i+1)G_i + 4a^2G_{i+2}
> \end{align*}
> $$

# Gaussian Overlaps

{: #def:cartesian-gaussian-overlap }
> **Definition (Cartesian Gaussian Overlap)**. 
> Let $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ be multi-indices,
> $a,b\in\mathbb{R}$ positive real numbers and
> $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$.
>
> The Cartesian Gaussian overlap is defined by:
>
> $$
> \Omega_{\boldsymbol{\alpha}\boldsymbol{\beta}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B}) :=
> G_\boldsymbol{\alpha}(\mathbf{r}, a, \mathbf{A})G_\boldsymbol{\beta}(\mathbf{r}, b, \mathbf{B})
> $$

Similarly to the Cartesian Gaussians, the overlaps also factorize along the
Cartesian coordinates into a product of one-dimensional overlaps.

{: #def:1d-cartesian-gaussian-overlap }
> **Definition (1d Cartesian Gaussian Overlap).**
> Let $i,j\in\mathbb{Z}$ be non-negative integers,
> $a,b\in\mathbb{R}$ positive real numbers and
> $A,B\in\mathbb{R}$.
>
> The one dimensional Cartesian Gaussian overlap is defined by:
>
> $$
> \Omega_{ij}(x, a, b, A, B) := G_i(x, a, A)G_j(x, b, B)
> $$

We can now factor the Cartesian Gaussian Overlap as:

{: #clm:cartesian-gaussian-overlap-factorization }
> **Claim (Cartesian Gaussian Overlap Factorization)**.
> Let $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}$ be multi-indices,
> $a,b\in\mathbb{R}$ positive real numbers and
> $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$.
>
> Then:
>
> $$
> \Omega_{\boldsymbol{\alpha}\boldsymbol{\beta}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B}) =
> \Omega_{\alpha_x\beta_x}(x, a, b, A_x, B_x)
> \Omega_{\alpha_y\beta_y}(y, a, b, A_y, B_y)
> \Omega_{\alpha_z\beta_z}(z, a, b, A_z, B_z)
> $$

One key reason that electron integrals of Cartesian Gaussians are easy to evaluate 
is that the product of two Gaussians is another Gaussian:

{: #clm:1d-gaussian-product-rule }
> **Claim (1d Gaussian Product Rule).** 
> Let $a,b\in\mathbb{R}$ be positive real numbers and
> $A,B\in\mathbb{R}$.
>
> Then:
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

The three-dimensional Cartesian Gaussian product rule follows immediately from
the [1d Gaussian Product Rule](#clm:1d-gaussian-product-rule)
and
[Cartesian Gaussian Overlap Factorization](#clm:cartesian-gaussian-overlap-factorization).

{: #clm:gaussian-product-rule }
> **Claim (Gaussian Product Rule).** 
> Let $a,b\in\mathbb{R}$ be positive real numbers and
> $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$.
>
> Then:
>
> $$
> G(\mathbf{r}, a, \mathbf{A})G(\mathbf{r}, b, \mathbf{B}) 
> = \mathbf{K}(a,b,\mathbf{A},\mathbf{B}) G(\mathbf{r}, p, \mathbf{P})
> $$
>
> Where $p$ and $\mu$ are defined by:
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
> **Claim (Overlap Vertical Transfer).** 
> Let $a,b\in\mathbb{R}$ be positive integers and $A,B\in\mathbb{R}$.
>
> For all non-negative $i\in\mathbb{Z}$ define:
>
> $$
> \Omega_i := \Omega_{i,0}(x, a, b, A, B)
> $$
>
> Then:
>
> 1. $(x - A)\Omega_i = \Omega_{i+1}$
> 2. $\frac{\partial}{\partial A}\Omega_i = 2a\Omega_{i+1} - i\Omega_{i-1}$
> 3. $\frac{\partial}{\partial B}\Omega_i = 2b\Omega_{i+1} + 2b(A-B) \Omega_i$

{: #clm:overlap-horizontal-transfer }
> **Claim (Overlap Horizontal Transfer).**
> Let $a,b\in\mathbb{R}$ positive real numbers and $A,B\in\mathbb{R}$.
>
> For all non-negative integers $i,j\in\mathbb{Z}$ define:
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

The Cartesian Gaussian $G_\boldsymbol{\alpha}(x, a, \mathbf{A})$
is the product of a polynomial in $x,y,z$ and an exponential function.
Hermite Gaussians are another way to generate products of polynomials and exponentials:

{: #def:hermite-gaussian }
> **Definition (Hermite Gaussian)** 
>Let $t,u,v\in\mathbb{N}$ be non-negative integers,
> $p\in\mathbb{R}$ a positive real number and $\mathbf{P}\in\mathbb{R}^3$. 
>
> The Hermite Gaussian is defined by:
>
> $$
> \Lambda_{tuv}(\mathbf{r}, p, \mathbf{P}) := 
> \left(\frac{\partial}{\partial P_x}\right)^t
> \left(\frac{\partial}{\partial P_y}\right)^u
> \left(\frac{\partial}{\partial P_z}\right)^v
> G(\mathbf{r}, p, \mathbf{P})
> $$

Similarly to Cartesian Gaussians, they can be factorized as 
one-dimensional Hermite Gaussians.

{: #def:1d-hermite-gaussian }
> **Definition (1d Hermite Gaussian)** 
>Let $t\in\mathbb{N}$ be a non-negative integer,
> $p\in\mathbb{R}$ a positive real number and $P\in\mathbb{R}$. 
>
> The one-dimensional Hermite Gaussian is defined by:
>
> $$
> \Lambda_t(x, p, P) := \left(\frac{\partial}{\partial P}\right)^t G(x, p, P)
> $$

{: #clm:hermite-gaussian-factorization }
> **Claim (Hermite Gaussian Factorization).**
> Let $t,u,v\in\mathbb{N}$ be non-negative integers,
> $p\in\mathbb{R}$ a positive real number and $\mathbf{P}\in\mathbb{R}^3$. 
>
> Then:
>
> $$
> \Lambda_{tuv}(\mathbf{r}, p, \mathbf{P}) = 
> \Lambda_t(x, p, P_x)
> \Lambda_u(y, p, P_y)
> \Lambda_v(z, p, P_z)
> $$

{: #clm:differentiation-product-bracket }
> **Claim (Differentiation Product Bracket).** 
> For every positive $t\in\mathbb{Z}$:
>
> $$
> [\frac{\partial^t}{\partial x^t}, x] = t \frac{\partial^{t-1}}{\partial x^{t-1}}
> $$

{: #clm:1d-hermite-gaussian-properties }
> **Claim (1d Hermite Gaussian Properties).**
> Let $p\in\mathbb{R}$ be a positive real number and $P\in\mathbb{R}$.
>
> For all non-negative integers $t\in\mathbb{Z}$ define:
>
> $$
> \Lambda_t := \Lambda_t(x, p, P)
> $$
>
> Then:
>
> 1. $(x - P)\Lambda_t = \frac{1}{2p}\Lambda_{t+1} + t\Lambda_{t-1}$
> 2. $\frac{\partial}{\partial P}\Lambda_t = \Lambda_{t+1}$

Note that by the 
[1d Gaussian Product Rule](#clm:1d-gaussian-product-rule)
the one-dimensional overlap $\Omega_{i,0}(x, a, b, A, B)$ is the
product of a degree $i$ polynomial in $x$ with $G(x, p, P)$ where
$p=a+b$ and $P = \frac{a}{p}A + \frac{b}{p}B$.

Similarly, $\Lambda_t(x, p, P)$ is the product of a degree $t$ polynomial in $x$
with $G(x, p, P)$. Therefore, it is possible to express $\Omega_{i,0}(x, a, b, A, B)$
as a linear combination of $\Lambda_0(x, p, P),\dots,\Lambda_i(x, p, P)$.

This is formalized in the following definition.
The coefficients of the linear combination will be determined recursively in the
following claim.

{: #def:1d-cartesian-overlap-to-hermite }
> **Definition (1d Cartesian Overlap To Hermite).**
> Let $i\in\mathbb{Z}$ be a non-negative
> integer, $a,b\in\mathbb{R}$ positive real numbers and $A,B\in\mathbb{R}$.
>
> Let $p=a+b$ and $P = \frac{a}{p}A + \frac{b}{p}B$.
>
> The coefficients $E^i_t(a, b, A, B)\in\mathbb{R}$ are defined to satisfy:
>
> $$
> \Omega_{i,0}(x, a, b, A, B) = \sum_{t=0}^i E^i_t(a, b, A, B)\Lambda_t(x, p, P)
> $$
>
> Furthermore, we define $E^i_t(a, b, A, B) = 0$ for all $t < 0$ or $ t > i$.

{: #clm:cartesian-overlap-to-hermite-recurrence }
> **Claim (Cartesian Overlap To Hermite Recurrence).**
> Let $i\in\mathbb{Z}$ be a non-negative integer, 
> $a,b\in\mathbb{R}$ positive real numbers and $A,B\in\mathbb{R}$.
>
> Let $p=a+b$ and $P=\frac{a}{p}A + \frac{b}{p}B$.
>
> Define:
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
[One Dimensional Hermite Gaussian Properties](#clm:one-dimensional-hermite-gaussian-properties)
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

Since Cartesian Gaussian Overlaps and Hermite Gaussians can be factored into their
one-dimensional analogs, we can extend
[1d Cartesian Overlap To Hermite](#def:1d-cartesian-overlap-to-hermite)
to three dimensions.

{: #def:cartesian-overlap-to-hermite }
> **Definition (Cartesian Overlap To Hermite).**
> Let $i,j,k,t,u,v\in\mathbb{Z}$ be non-negative integers,
> $a,b\in\mathbb{R}$ positive real numbers and $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$.
>
> The coefficients $E_{tuv}^{ijk}(a, b, \mathbf{A}, \mathbf{B})\in\mathbb{R}$ are defined by:
>
> $$
> E_{tuv}^{ijk}(a, b, \mathbf{A}, \mathbf{B}) :=
> E_t^i(a, b, A_x, B_x)
> E_u^j(a, b, A_y, B_y)
> E_v^k(a, b, A_z, B_z)
> $$

{: #clm:cartesian-overlap-to-hermite }
> **Claim (Cartesian Overlap To Hermite).**
> Let $i,j,k,t,u,v\in\mathbb{Z}$ be non-negative integers,
> $a,b\in\mathbb{R}$ positive real numbers and $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$.
>
> Let $p=a+b$ and $\mathbf{P}=\frac{a}{p}\mathbf{A} + \frac{b}{p}\mathbf{B}$.
>
> Then:
>
> $$
> \Omega_{(ijk),\mathbf{0}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B}) =
> \sum_{tuv=0}^{ijk}E_{tuv}^{ijk}(a, b, \mathbf{A}, \mathbf{B})
> \Lambda_{tuv}^{ijk}(\mathbf{r}, p \mathbf{P})
> $$


# Overlap Integrals

The simplest type of electron integral is the _overlap integral_:

> **Definition (Overlap Integral).** 
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$ and
> $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ multi-indices.
>
> The overlap integral is defined as:
>
> $$
> S_{\boldsymbol{\alpha}\boldsymbol{\beta}}(a,b,\mathbf{A},\mathbf{B}) 
> = \int_{\mathbb{R}^3} G_{\boldsymbol{\alpha}}(\mathbf{r}, a, \mathbf{A}) G_{\boldsymbol{\beta}}(\mathbf{r}, b, \mathbf{B}) d\mathbf{r}
> $$

By [Cartesian Gaussian Factorization](#clm:cartesian-gaussian-factorization),
Cartesian Gaussians can be factorized into one-dimensional Cartesian Gaussians.
Therefore, the overlap integral can be factorized as well.

> **Definition (1d Overlap Integral).** 
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $A,B\in\mathbb{R}$ and
> $i,j\in\mathbb{Z}$ non-negative integers.
>
> The one-dimensional overlap integral is defined as:
>
> $$
> S_{ij}(a,b,A,B) 
> = \int_{-\infty}^{\infty} 
> G_{i}(x, a, A) G_{j}(x, b, B) dx
> $$

> **Claim (Overlap Integral Factorization).** 
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$ and
> $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ multi-indices.
>
> Then:
>
> $$
> S_{\boldsymbol{\alpha}\boldsymbol{\beta}}(a,b,\mathbf{A},\mathbf{B}) =
> S_{\alpha_x\beta_x}(a, b, A_x, B_x)
> S_{\alpha_y\beta_y}(a, b, A_y, B_y)
> S_{\alpha_z\beta_z}(a, b, A_z, B_z)
> $$

Therefore, it suffices to develop a method for computing one-dimensional overlap
integrals.

The goal of this section is to prove the following set of recurrence relations which
determine $S_{ij}(a, b, A, B)$ for all non-negative $i,j\in\mathbb{Z}$

> **Claim (Overlap Integral Base Case).**
> Let $a,b\in\mathbb{R}$ be positive real numbers and
> $A,B\in\mathbb{R}$.
>
> Then:
>
> $$
> S_{0,0}(a,b,A,B) 
> = \sqrt{\frac{\pi}{p}} K(a,b,A,B) 
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">
This follows from the [1d Gaussian Product Rule](#clm:1d-gaussian-product-rule)
and [1-d Spherical Gaussian Integral](#clm:1d-spherical-gaussian-integral).

_q.e.d_
</div>
</details>

> **Claim (Overlap Integral Recurrence).**
> Let $a,b\in\mathbb{R}$ be positive real numbers and
> $A,B\in\mathbb{R}$.
>
> Let $p=a+b$ and $P=\frac{a}{p}A + \frac{b}{p}B$.
>
> For non-negative integers $i,j\in\mathbb{Z}$ define:
>
> $$
> S_{ij} := S_{ij}(a, b, A, B)
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

By definitions [Cartesian Gaussian Overlap](#def:cartesian-gaussian-overlap)
and [Cartesian Overlap To Hermite](#def:cartesian-overlap-to-hermite):

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

By [Spherical Gaussian Integral](#clm:spherical-gaussian-integral):

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
[One Dimensional Cartesian Overlap To Hermite Recurrence](#clm:one-dimensional-cartesian-overlap-to-hermite-recurrence)
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

Recall that the [Laplace operator](https://en.wikipedia.org/wiki/Laplace_operator) 
$\nabla^2$ is defined by:

$$
\nabla^2 := \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + 
\frac{\partial^2}{\partial z^2}
$$

> **Definition (Kinetic Energy Integral).**
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$ and
> $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ be multi-indices.
>
> The kinetic energy integral is defined by:
>
> $$
> T_{\boldsymbol{\alpha},\boldsymbol{\beta}}(a, b, \mathbf{A},\mathbf{B}) :=
> \int_{\mathbb{R}^3} 
> G_{\boldsymbol{\alpha}}(\mathbf{r}, a, \mathbf{A}) 
> \nabla^2
> G_{\boldsymbol{\beta}}(\mathbf{r}, b, \mathbf{B}) 
> d\mathbf{r}
> $$

Similarly to overlap integrals, we can decompose this integral over $\mathbb{R}^3$
into simpler one-dimensional integrals over $\mathbb{R}$.

We'll start by defining the one-dimensional kinetic energy integral.

{: #def:1d-kinetic-energy-integral }
> **Definition (1d Kinetic Energy Integral).**
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $A,B\in\mathbb{R}$ and
> $i,j\in\mathbb{Z}$ non-negative integers.
>
> The one dimensional kinetic energy integral is defined by:
>
> $$
> T_{ij}(a, b, A, B) :=
> \int_0^\infty
> G_i(x, a, A) 
> \frac{\partial^2}{\partial^2 x}
> G_j(x, b, B) 
> dx
> $$

> **Claim (Kinetic Energy Integral Reduction).**
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B}\in\mathbb{R}^3$ and
> $\boldsymbol{\alpha},\boldsymbol{\beta}\in\mathbb{Z}^3$ be multi-indices.
>
> Define the following one-dimensional overlap and kinetic energy integrals:
>
> $$
> \begin{alignat*}{2}
> S_{\alpha_x\beta_x} &:= S_{\alpha_x\beta_x}(a, b, A_x, B_x) &\qquad&  T_{\alpha_x\beta_x} &:= T_{\alpha_x\beta_x}(a, b, A_x, B_x) \\
> S_{\alpha_y\beta_y} &:= S_{\alpha_y\beta_y}(a, b, A_y, B_y) &\qquad& T_{\alpha_y\beta_y} &:= T_{\alpha_y\beta_y}(a, b, A_y, B_y) \\
> S_{\alpha_z\beta_z} &:= S_{\alpha_z\beta_z}(a, b, A_z, B_z) &\qquad& T_{\alpha_z\beta_z} &:= T_{\alpha_z\beta_z}(a, b, A_z, B_z)
> \end{alignat*}
> $$
>
> Then:
>
> $$
> T_{\boldsymbol{\alpha},\boldsymbol{\beta}}(a, b, \mathbf{A},\mathbf{B}) =
> T_{\alpha_x\beta_x}S_{\alpha_y\beta_y}S_{\alpha_z\beta_z} +
> S_{\alpha_x\beta_x}T_{\alpha_y\beta_y}S_{\alpha_z\beta_z} +
> S_{\alpha_x\beta_x}S_{\alpha_y\beta_y}T_{\alpha_z\beta_z}
> $$

The next claim shows how to compute the one-dimensional kinetic energy integrals
in terms of one-dimensional overlap integrals.

> **Claim (1d Kinetic Energy From Overlaps).**
> Let $a,b\in\mathbb{R}$ be positive real numbers and
> $A,B\in\mathbb{R}$.
>
> For all non-negative integers $i,j\in\mathbb{Z}$ define:
>
> $$
> S_{ij} = S_{ij}(a, b, A, B)
> $$
>
> Then:
>
> $$
> T_{ij}(a, b, A, B) =
> j(j-1)S_{i,j-2} - 
> 2b(2j + 1)S_{ij} +
> 4b^2 S_{i,j+2}
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The claim follows immediately from the definition of the kinetic energy
integral and
[1d Cartesian Gaussian Derivatives](#clm:1d-cartesian-gaussian-derivatives)
</div>
</details>

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


First we'll assume that $\boldsymbol{\beta}=\mathbf{0}$ and focus on finding recurrence
relations for $\boldsymbol{\alpha}=(i,j,k)$.

As usual, set $p=a+b$ and $\mathbf{P} = \frac{a}{p}\mathbf{A} + \frac{b}{p}\mathbf{B}$.

By definitions [Cartesian Gaussian Overlap](#def:cartesian-gaussian-overlap)
and [Cartesian Overlap To Hermite](#def:cartesian-overlap-to-hermite):

$$
\begin{equation}\label{eq:one-electron-expansion}
\begin{aligned}
I_{(i,j,k),\mathbf{0}}(a, b, \mathbf{A}, \mathbf{B}, \mathbf{C}) &=
\int_{\mathbb{R}^3}\frac{\Omega_{(i,j,k),\mathbf{0}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B})}{||\mathbf{r}-\mathbf{C}||}d\mathbf{r} \\
&= \sum_{t,u,v=0}^{ijk}E^{ijk}_{tuv}(a, b, \mathbf{A}, \mathbf{B})
\int_{\mathbb{R}^3}\frac{\Lambda_{tuv}(\mathbf{r},p,\mathbf{P})}{||\mathbf{r}-\mathbf{C}||}d\mathbf{r} \\
&= \sum_{t,u,v=0}^{ijk}E^{ijk}_{tuv}(a, b, \mathbf{A}, \mathbf{B})
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
\int_{\mathbb{R}^3}\frac{G_\mathbf{0}(\mathbf{r},p,\mathbf{P})}{||\mathbf{r}-\mathbf{C}||}d\mathbf{r}
\end{aligned}
\end{equation}
$$

The point of this reformulation is that integrand no longer contains the 
multi-index $\boldsymbol{\alpha}=(i,j,k)$ and can be solved in closed form using the
[Boys function](https://molssi.github.io/MIRP/boys_function.html).

{: #def:boys-function }
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
[Spherical Gaussian Integral](#clm:spherical-gaussian-integral).

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

Integrating over $\mathbf{r}$ and applying
[Spherical Gaussian Integral](#clm:spherical-gaussian-integral):

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

{: #def:boys-partial-derivatives }
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

{: #def:nth-order-coulomb-integral }
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
relations for the [Boys Partial Derivatives](#def:boys-partial-derivatives) $R_{tuv}^n$.
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
>    V_{000}^n = K(a,b,\mathbf{A},\mathbf{B})F_n(s||\mathbf{P} - \mathbf{C}||)
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
[One Electron Coulomb Integral](#def:one-electron-coulomb-integral)

> **Claim (One Electron Coulomb Base Case).**
> Let $a,b\in\mathbb{R}$ be positive real numbers,
> $\mathbf{A},\mathbf{B},\mathbf{C}\in\mathbb{R}^3$ and
> $i,j,k\in\mathbb{Z}$ non-negative integers. Then:
>
> $$
> I_{(i,j,k), \mathbf{0}}(a, b, \mathbf{A},\mathbf{B},\mathbf{C}) =
> \frac{2\pi}{a+b}V_{ijk}^0(a, b, a+b, \mathbf{A},\mathbf{B},\mathbf{C})
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
[One Electron Coulomb Integral](#def:one-electron-coulomb-integral) and
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
> G_\boldsymbol{\alpha}(\mathbf{r}_1, a, \mathbf{A})
> G_\boldsymbol{\beta}(\mathbf{r}_1, b, \mathbf{B})
> G_\boldsymbol{\gamma}(\mathbf{r}_2, c, \mathbf{C})
> G_\boldsymbol{\delta}(\mathbf{r}_2, d, \mathbf{D})
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

Substituting the definition of the [Boys Function](#def:boys-function) gives:

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
[Spherical Gaussian Integral](#clm:spherical-gaussian-integral):

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

which, by the definition of the [Boys Function](#def:boys-function), is equal to:

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
J_{ijk} := 
J_{(i,j,k),\mathbf{0},\mathbf{0},\mathbf{0}}
(a, b, c, d, \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D})
$$

By the [Gaussian Product Rule](#clm:gaussian-product-rule):

$$
G_\mathbf{0}(\mathbf{r}, c, \mathbf{C})
G_\mathbf{0}(\mathbf{r}, d, \mathbf{D}) =
K(c, d, \mathbf{C}, \mathbf{D})G_\mathbf{0}(\mathbf{r}, q, \mathbf{Q})
$$

By the definition of the
[Cartesian Gaussian Overlap](#def:cartesian-gaussian-overlap)
we can rewrite the two-electron integral as:

$$
J_{ijk} = 
K(c, d, \mathbf{C}, \mathbf{D})
\int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
\frac{
\Omega_{(i,j,k), \mathbf{0}}(\mathbf{r}_1, a, b, \mathbf{A}, \mathbf{B})
G_\mathrm{0}(\mathbf{r}_2, q, \mathbf{Q})
}{||\mathbf{r}_1 - \mathbf{r}_2||}
d\mathbf{r}_1d\mathbf{r}_2
$$

We'll now use 
[Cartesian Overlap To Hermite](#def:cartesian-overlap-to-hermite)
to expand 
$\Omega_{(i,j,k), \mathbf{0}}(\mathbf{r}, a, b, \mathbf{A}, \mathbf{B})$
in terms of Hermite Gaussians:

$$
J_{ijk} = 
K(c, d, \mathbf{C}, \mathbf{D})
\sum_{tuv}^{ijk}
E_{tuv}^{ijk}(a, b, \mathbf{A}, \mathbf{B})
\int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
\frac{
\Lambda_{tuv}(\mathbf{r}_1, p, \mathbf{P})
G_\mathrm{0}(\mathbf{r}_2, q, \mathbf{Q})
}{||\mathbf{r}_1 - \mathbf{r}_2||}
d\mathbf{r}_1d\mathbf{r}_2
$$

By the definition of
[Hermite Gaussians](#def:hermite-gaussian):

$$
J_{ijk} = 
K(c, d, \mathbf{C}, \mathbf{D})
\sum_{tuv}^{ijk}
E_{tuv}^{ijk}(a, b, \mathbf{A}, \mathbf{B})
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
\int_{\mathbb{R}^3}\int_{\mathbb{R}^3}
\frac{
G_\mathbf{0}(\mathbf{r}_1, p, \mathbf{P})
G_\mathbf{0}(\mathbf{r}_2, q, \mathbf{Q})
}{||\mathbf{r}_1 - \mathbf{r}_2||}
d\mathbf{r}_1d\mathbf{r}_2
$$

Applying the 
[Spherical Two Electron Coulomb Integral](#clm:spherical-two-electron-coulomb-integral):

$$
J_{ijk} = 
\frac{2\pi^{5/2}}{pq\sqrt{p+q}}
K(c, d, \mathbf{C}, \mathbf{D})
\sum_{tuv}^{ijk}
E_{tuv}^{ijk}(a, b, \mathbf{A}, \mathbf{B})
\left(\frac{\partial}{\partial P_x}\right)^t
\left(\frac{\partial}{\partial P_y}\right)^u
\left(\frac{\partial}{\partial P_z}\right)^v
F_0(s||\mathbf{P} - \mathbf{Q}||^2)
$$

The claim now follows from the definition of the
[$n$-th order Coulomb intergal](#def:nth-order-coulomb-integral).

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

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

This follows from applying 
[Cartesian Gaussian Overlap Factorization](#clm:cartesian-gaussian-overlap-factorization)
and
[Overlap Horizontal Transfer](#clm:overlap-horizontal-transfer)
to the first two and second two pairs of Cartesian Gaussians in the definition
of the
[Two Electron Coulomb Integral](#def:two-electron-coulomb-integral).

_q.e.d_

</div>
</details>

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
[Two Electron Coulomb Integral](#def:two-electron-coulomb-integral) is defined as an
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
\mathbf{e} \cdot \nabla J_{ij}(A_x,B_x,C_x,D_x) = 0
$$

By the definition of the gradient this implies:

$$
\left(\frac{\partial}{\partial A_x} + 
 \frac{\partial}{\partial B_x} + 
 \frac{\partial}{\partial C_x} + 
 \frac{\partial}{\partial D_x}\right)
 J_{ij}(A_x,B_x,C_x,D_x) = 0
$$

By using the definition of the 
[Two Electron Coulomb Integral](#def:two-electron-coulomb-integral),
[1d Cartesian Gaussian Derivatives](#clm:1d-cartesian-gaussian-derivatives)
and
[Two Electron Coulomb Electron Transfer](#clm:two-electron-coulomb-electron-transfer):

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
