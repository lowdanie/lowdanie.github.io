---
layout: post
title: "The Hidden Subgroup Problem"
date: 2025-04-23
mathjax: true
utterance-issue: 12
---

# Introduction

A fundamental challenge in
[quantum computing](https://en.wikipedia.org/wiki/Quantum_computing) is to find
problems for which there is a
[quantum algorithm](https://en.wikipedia.org/wiki/Quantum_algorithm) that is
significantly (i.e more than polynomially) faster than the best known classical
algorithm.

One of the most impactful examples of this is
[Shor's Algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm) for
[integer factorization](https://en.wikipedia.org/wiki/Integer_factorization).
The significance of this problem is that the ability to efficiently factor large
integers would break the ubiquitous RSA cryptography scheme.

The possibility of a quantum computer that can run Shor's algorithm has led to
the development of alternative cryptographic schemes known as
[Post Quantum Cryptography](https://en.wikipedia.org/wiki/Post-quantum_cryptography)
which are considered resistant to quantum computers. Rather than relying on the
difficulty of integer factorization, the leading PQC schemes are forms of
[Lattice Cryptography](https://en.wikipedia.org/wiki/Lattice-based_cryptography)
which rely on the difficulty of a lattice problem called
[Shortest Vector Problem](<https://en.wikipedia.org/wiki/Lattice_problem#Shortest_vector_problem_(SVP)>)
(SVP)

It turns out that integer factorization can be reformulated as a special case of
a more general problem called the
[Hidden Subgroup Problem](https://en.wikipedia.org/wiki/Hidden_subgroup_problem).
Furthermore, Shor's algorithm can be generalized to an efficient quantum
algorithm for the HSP whenever the group in question is
[commutative](https://en.wikipedia.org/wiki/Abelian_group).

Interestingly, almost all known problems with a significant quantum speedup are
instances of the HSP for commutative groups.

Furthermore, even the Shortest Vector Problem is an instance of the HSP for a
certain _non-commutative_ group. The reason that the SVP is considered resistant
to quantum algorithms is that it is not known how to extend Shor's algorithm for
the HSP from commutative to non-commutative groups.

In this post we'll introduce and motivate Shor's algorithm for the commutative
Hidden Subgroup Problem via a sequence of examples. We'll then see how the
Shortest Vector Problem is an instance of the HSP for a non-commutative group
called the [Dihedral Group](https://en.wikipedia.org/wiki/Dihedral_group) and
discuss the difficulty of extending Shor's algorithm that group.

# The Group With Two Elements

Consider the group with two elements $\mathbb{Z}/2\mathbb{Z}$. This group has
two elements $\\{0,1\\}$ and addition is defined modulo $2$. We'll denote the
addition operation by the symbol $\oplus$.

The _Hidden Subgroup Problem_ for $\mathbb{Z}/2\mathbb{Z}$ can be stated as
follows:

> Let $A=\\{a_0,a_1\\}$ be a set with two elements and
> $f:\mathbb{Z}/2\mathbb{Z}\rightarrow A$ a function from $\mathbb{Z}$ to $A$.
> Determine whether $f(0)\in A$ is equal to $f(1)\in A$.

Of course, it is straightforward to solve this problem classically by simply
calculating $f(0)$ and $f(1)$ and comparing the results. But, given that $f$ may
be expensive to compute, it's interesting to wonder if there exists a quantum
algorithm for this problem that only evaluates $f$ _once_.

A natural starting point is to create a superposition over the elements of
$\mathbb{Z}/2\mathbb{Z}$ together with an extra bit of scratch space:

$$
|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle|0\rangle + |1\rangle|0\rangle)
$$

In quantum computing terminology, the extra bit at the end is referred to as the
[ancilla bit](https://en.wikipedia.org/wiki/Ancilla_bit).

We can then apply $f$ to $|\psi\rangle$ and store the result in the ancilla bit
to obtain:

$$
f(|\psi\rangle) = \frac{1}{\sqrt{2}}(|0\rangle|f(0)\rangle + |1\rangle|f(1)\rangle)
$$

If we measure the ancilla bit there are two possible results. If
$f(0)=f(1)=a_{\mathrm{eq}}$ then the result of the measurement will be the
common output $a_{\mathrm{eq}}$ and the new state will be:

$$
|\psi_{=}\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|a_{\mathrm{eq}}\rangle
$$

On the other hand, if $f(0)\neq f(1)$ then the result of the measurement will
either be $f(0)$ or $f(1)$ with equal probability. If the result of the
measurement is $f(0)$ then the new state will be:

$$
|\psi_0\rangle = |0\rangle|f(0)\rangle
$$

and if the result is $f(1)$ then the new state will be:

$$
|\psi_1\rangle = |1\rangle|f(1)\rangle
$$

We will denote by $|\psi\_{\neq}\rangle$ the
[mixed state](https://en.wikipedia.org/wiki/Quantum_state#Mixed_states) that is
equally likely to be one of $|\psi_0\rangle$ or $|\psi_1\rangle$.

At first glance it would appear as though we've solved the HSP with only a
single application of $f$ to $|\psi\rangle$ followed by a measurement. However,
we still need a way to distinguish between $|\psi_{=}}\rangle$ and
$|\psi_{\neq}\rangle$.

What happens if we just measure the final state?

The outcome of measuring $\|\psi\_{=}\rangle$ will be either
$(0, a\_{\mathrm{eq}})$ or $(1, a\_{\mathrm{eq}})$ with equal likelihood.
Without knowing any additional information about $f$, $a\_{\mathrm{eq}}$ is
equally likely be equal to either $a\_0$ or $a\_1$. Therefore, the result of
measuring $\|\psi\_{=}\rangle$ is equally likely to be any of the four pairs
$(i, a\_j)$ where $i,j\in\\{0,1\\}$.

On the other hand, the result of measuring $\|\psi_{\neq}\rangle$ is equally
likely to be $(0, f(0))$ or $(1, f(1))$. Without any prior information about
$f$, this is also equally likely to be $(i, a_0)$ or $(i, a_1)$. Therefore,
assuming that $f(0)\neq f(1)$, the final result is also equally likely to be any
of the four pairs $(i, a\_j)$ where $i,j\in\\{0,1\\}$.

In summary, measuring the final state in the standard basis provides no
additional information about $f$.

How else can we distinguish between $|\psi_{=}\rangle$ and $\psi\_{\neq}\rangle?

Note that, up to a scaling factor, $|\psi_{=}\rangle$ is equal the sum of the
standard basis vectors $|0\range$ and $|1\rangle$. On the other hand
$\psi_{\neq}]rangle$ is equal to either the basis vector $|0\rangle$ or the
basis vector $|1\rangle$. Therefore, all we need is a way to determine whether
our final state is a sum of two basis vectors or just equal to one of them.

The key idea is to perform a unitary change of basis that will cause the sum of
the standard basis vectors to partially cancel out. Specifically, consider the
transformation $H$ which is defined by:

$$
\begin{align*}
H |0\rangle &= \frac{1}{\sqrt{2}}\left(|0\rangle + |1\rangle\right) \\
H |1\rangle &= \frac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle\right)
\end{align*}
$$

The point of $H$ is that if we apply it to the sum
$\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ then the vector $|1\rangle$ cancels
out and we get:

$$
H\left(\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)\right) = |0\rangle
$$

Let's see what happens if we apply $H$ to our states $|\psi_{=}\rangle$ and
$|\psi_{\neq}\rangle$.

# Simon's Problem

Simon's problem was introduced by
[Daniel Simon](https://epubs.siam.org/doi/10.1137/S0097539796298637) in 1994 as
an example of an problem with an efficient quantum algorithm.

Let $B_N = \\{0,1\\}^N$ denote the set of bit-strings of length $N$ and let
$\oplus$ denote the bitwise XOR operation.

For example, if $N=3$ then $110$ and $101$ are elements of $B_3$ and:

$$
110 \oplus 101 = 011
$$

Simon's problem is defined as follow:

> We are given access to a black-box for computing a function
> $f: B_N \rightarrow A$ from $B_N$ to a set $A$. In addition, we are promised
> that there is a secret bit-string $\mathbf{s}\in G$ such that
> $f(\mathbf{x}) = f(\mathbf{x}')$ if and only if $\mathbf{x}' = \mathbf{x}$ or
> $\mathbf{x}' = \mathbf{x} \oplus \mathbf{s}$. The challenge is to find $s$
> with as few calls as possible to the black box for $f$.

To get some intuition for the role of the secret bit-string $\mathbf{s}$,
consider the following function on $B_3$ where $A = B_2$:

| $\mathbf{x}$ | $f(\mathbf{x})$ |
| ------------ | --------------- |
| $000$        | $10$            |
| $001$        | $00$            |
| $010$        | $11$            |
| $011$        | $01$            |
| $100$        | $00$            |
| $101$        | $10$            |
| $110$        | $01$            |
| $111$        | $11$            |

It is easy to check that in this case, the solution to Simon's problem is
$\mathbf{s}=101$. Note that $f$ is a 2 to 1 function. In general, we can think
of $\mathbf{s}$ as a bitmask where $f(\mathbf{x}) = f(\mathbf{x}')$ if and only
if $\mathbf{x}$ and $\mathbf{x}'$ differ exactly at the elements masked by
$\mathbf{s}$.

How hard is it to solve Simon's problem classically? Without knowing anything
about $f$, we intuitively have to keep calculating $f$ for different elements
$\mathbf{x}\in B_N$ until we get the same result twice. Based on the analysis of
the
[Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem#Arbitrary_number_of_days),
we have to compute $f$ for approximately $\sqrt{|B_N|}=2^{N/2}$ elements in
order to have a good chance of finding a collisions.

Simon's algorithm, introduced in the
[same paper](https://epubs.siam.org/doi/10.1137/S0097539796298637), is a quantum
algorithm that solves Simon's problem in $\mathcal{O(N)}$ time - an exponential
improvement over the best classical algorithm.

The first step of the algorithm os to prepare a superposition of all elements in
$B_N$, together with some auxiliary qubits that will be used to compute $f$:

$$
|\psi\rangle = \sum_{\mathbf{x}\in B_N}|\mathbf{x}\rangle |\mathbf{0}\rangle
$$

The auxiliary bits at the end are commonly referred to as the
[ancilla bits](https://en.wikipedia.org/wiki/Ancilla_bit).

We can then apply $f$ to obtain:

$$
\sum_{\mathbf{x}\in B_N}|\mathbf{x}\rangle |f(\mathbf{x})\rangle
$$

Note that if we now measure the ancilla bits we will observe some value
$f(\mathbf{x})\in A$ and the new state will be:

$$
|\varphi_\mathbf{x}\rangle = \sum_{\mathbf{x}'\in B_N,\, f(\mathbf{x}')=f(\mathbf{x})}|\mathbf{x}'\rangle |a\rangle
$$

In other words, we obtain a superposition of the elements in $B_N$ that are
mapped to $f(\mathbf{x})$ by $f$. By our assumption on $f$, the only elements
that get mapped to $f(\mathbf{x})$ are $\mathbf{x}$ itself and
$\mathbf{x}' = \mathbf{x}\oplus\mathbf{s}$ where $\mathbf{s}$ is the hidden
element. Therefore, after discarding the ancilla bits we can rewrite
$|\varphi_\mathbf{x}\rangle$ as:

$$
|\varphi_\mathbf{x}\rangle = |\mathbf{x}\rangle + |\mathbf{x}\oplus\mathbf{s}\rangle
$$

At a first glance it seems like we can easily deduce $\mathbf{s}$ from
$|\varphi_\mathbf{x}\rangle$ since
$\mathbf{s} = \mathbf{x} \oplus (\mathbf{x}\oplus\mathbf{s})$. The fundamental
issue is that we cannot directly observe both of the basis vectors in
$|\varphi_\mathbf{x}\rangle$. Specifically, if we measure
$|\varphi_\mathbf{x}\rangle$ then we will observe _either_ $\mathbf{x}$ _or_
$\mathbf{x}\oplus\mathbf{s}$ with equal probability. Without knowing
$\mathbf{x}$, neither of these values provide any information whatsoever about
$\mathbf{s}$.

To get another perspective on $|\varphi\_\mathbf{x}\rangle$, consider the _Shift
Operator_

$$
\begin{align*}
L_\mathbf{x}: \mathbb{C}[B_N] &\rightarrow \mathbb{C}[B_N] \\
|\mathbf{x}'\rangle &\mapsto |\mathbf{x}\oplus\mathbf{x}'\rangle
\end{align*}
$$

We are using $\mathbb{C}[B_N]$ to denote the complex vector space with an
orthogonal basis given by $|\mathbf{x}'\rangle$ for $\mathbf{x}'\in B_N$. The
shift operator $L_\mathbf{x}$ sends the basis element
$|\mathbf{x}'\rangle\in\mathbb{C}[B_N]$ to
$|\mathbf{x}\oplus\mathbf{x}'\rangle \in \mathbb{C}[B_N]$. Using $L_\mathbb{x}$
we can rewrite $|\varphi_\mathbb{x}\rangle$ as:

$$
|\varphi_\mathbb{x}\rangle = L_\mathbf{x}(|\mathbf{0}\rangle + |\mathbf{s}\rangle)
$$

With this perspective, we could recover information about $\mathbf{s}$ from
$|\varphi_\mathbb{x}\rangle$ if we had some way of undoing the shift operator
$L_\mathbf{x}$ to obtain $|\mathbf{0}\rangle + |\mathbf{s}\rangle$. The key idea
of Simon's algorithm is to apply a change of basis to
$|\varphi\_\mathbf{x}\rangle$ before measuring it. The new basis will be one
that _simultaneously diagonalizes_ the shift operators $L_\mathbf{x}$ for all
$\mathbf{x}\in B_N$.

Since $L_\mathbf{x}$ is diagonalized in this basis for all $\mathbb{x}$,
measuring
$|\varphi_\mathbb{x}\rangle = L_\mathbf{x}(|\mathbf{0}\rangle + |\mathbf{s}\rangle)$
in the new basis is equivalent to measuring
$|\mathbf{0}\rangle + |\mathbf{s}\rangle$. Therefore, measuring in the new basis
will provide direct information about $\mathbb{s}$.

In the following sections we will construct this new basis and show how to use
it to recover $\mathbf{s}$.

## Diagonalizing The Shift operators

Let $\mathbf{x}\in B_N$ be a bit-string. In the previous section we defined the
shift operator $L_\mathbf{x}$ on the Hilbert space $\mathbb{C}[B_N]$:

$$
\begin{align*}
L_\mathbf{x}: \mathbb{C}[B_N] &\rightarrow \mathbb{C}[B_N] \\
|\mathbf{x}'\rangle &\mapsto |\mathbf{x}\oplus\mathbf{x}'\rangle
\end{align*}
$$

For example, if $N=3$ and $\mathbf{x}=110$ then:

$$
L_\mathbf{x}(|010\rangle + |111\rangle) = |100\rangle + |001\rangle
$$

Note that clearly $L_\mathbf{x}$ is not diagonal in the standard basis of
$\mathbb{C}[B_N]$ given by elements of $B_N$. For example, applying $L_{110}$ to
$|101\rangle$ gives:

$$
L_{110}(|101\rangle) = |011\rangle
$$

which is not a scalar multiple of the input $|101\rangle$.

The goal of this section is to find another basis of $\mathbb{C}[B_N]$ which
diagonalizes $L_\mathbf{x}$ for all $\mathbf{x}$.

Suppose $|\varphi\rangle\in\mathbb{C}[B_N]$ is a vector in the new basis. This
means that, for all $\mathbf{x}\in B_N$, $|\varphi\rangle$ is an eigenvector of
$L_\mathbf{x}$ for all $\mathbf{x}\in B_N$ with some eigenvalue
$\lambda(\mathbf{x})$. The eigenvalues $\lambda(\mathbf{x})$ can be packaged
into a function

$$
\lambda: B_N \rightarrow \mathbb{C}
$$

We'll start with the following observation about $\lambda$:

> **Claim.** Suppose that $|\varphi\rangle\in\mathbb{C}[B_N]$ is a vector
> satisfying $L_\mathbf{x}|\varphi\rangle = \lambda(\mathbf{x})|\varphi\rangle$
> for all $\mathbf{x}\in B_N$. Then for all $\mathbf{x},\mathbf{x}'\in B_N$:
>
> $$
> \lambda(\mathbf{x}\oplus\mathbf{x}') = \lambda(\mathbf{x}) \cdot \lambda(\mathbf{x}')
> $$

_Proof._ Since addition in $B_N$ is associative, the shift operators satisfy:

$$
\begin{equation}\label{eq:shift-composition-old}
L_\mathbf{x} \circ L_{\mathbf{x}'} = L_{\mathbf{x}\oplus\mathbf{x}'}
\end{equation}
$$

Applying the left hand side of \ref{eq:shift-composition} to $|\varphi\rangle$
gives:

$$
\begin{align*}
L_\mathbf{x} \circ L_{\mathbf{x}'}|\varphi\rangle &= L_\mathbf{x}(\lambda(\mathbf{x}')|\varphi\rangle) \\
&= \lambda(\mathbf{x})\cdot\lambda(\mathbf{x}')|\varphi\rangle
\end{align*}
$$

Applying the right hand side of \ref{eq:shift-composition} to $|\varphi\rangle$
gives:

$$
L_{\mathbf{x}\oplus\mathbf{x}'}|\varphi\rangle = \lambda(\mathbf{x}\oplus\mathbf{x}')|\varphi\rangle
$$

Together this implies:

$$
\lambda(\mathbf{x})\cdot\lambda(\mathbf{x}')|\varphi\rangle =
\lambda(\mathbf{x}\oplus\mathbf{x}')|\varphi\rangle
$$

and the claim follows.

_q.e.d_

Functions satisfying this type of multiplicative property are called
[characters](<https://en.wikipedia.org/wiki/Character_(mathematics)>). More
precisely:

> **Definition (Character)** Let $G$ be a group. A _character_ on $G$ is a
> non-zero complex function
>
> $$
> \chi: G \rightarrow \mathbb{C}
> $$
>
> satisfying
>
> $$
> \chi(g_1 g_2) = \chi(g_1)\chi(g_2)
> $$
>
> for all $g_1,g_2\in G$.

We can restate the claim above as saying that the eigenvalue function $\lambda$
must be a character on $B_N$. Interestingly, implies that we can use the
eigenvalues $\lambda(\mathbf{x})$ to construct the corresponding eigenvector
$|\varphi\rangle$:

{: #clm:eig-from-char-old }

> **Claim (Eigenvectors From Characters).** Let
> $\chi: B_N \rightarrow \mathbb{C}$ be a character on $B_N$. Define
> $|\varphi\rangle\in\mathbb{C}[B_N]$ by:
>
> $$
> |\varphi\rangle = \sum_{\mathbf{x}}\chi(\mathbf{x})|\mathbf{x}\rangle
> $$
>
> Then, all $\mathbf{x}\in B_N$, $|\varphi\rangle$ is an eigenvector of
> $L_\mathbf{x}$ with eigenvalue $\chi(\mathbf{x})$.

_Proof._ Direct calculation shows that:

$$
\begin{align*}
L_{\mathbf{x}}|\varphi\rangle &= L_{\mathbf{x}}\sum_{\mathbf{x}'}\chi(\mathbf{x}')|\mathbf{x}'\rangle \\
&= \sum_{\mathbf{x}'}\chi(\mathbf{x}')|\mathbf{x}\oplus\mathbf{x}'\rangle \\
&= \sum_{\mathbf{x}'}\chi(\mathbf{x}\oplus\mathbf{x}')|\mathbf{x}'\rangle \\
&= \sum_{\mathbf{x}'}\chi(\mathbf{x})\chi(\mathbf{x}')|\mathbf{x}'\rangle \\
&= \chi(\mathbf{x})\cdot\sum_{\mathbf{x}'}\chi(\mathbf{x}')|\mathbf{x}'\rangle \\
&= \chi(\mathbf{x})|\varphi\rangle
\end{align*}
$$

_q.e.d_

This reduces the problem of simultaneously diagonalizing $L_\mathbf{x}$ to
finding characters of $B_N$.

To facilitate notation, we'll denote the bitwise dot product modulo $2$ on two
elements $\mathbf{x},\mathbf{y}\in B_N$ by $\mathbf{x}\cdot\mathbf{y}$.
Specifically:

$$
\mathbf{x} \cdot \mathbf{y} = \sum_{i=1}^N x_i y_i \mathrm{(mod 2)}
$$

For each $\mathbf{y}\in B_N$ we can define a character $\chi_{\mathbf{y}}$ on
$B_N$ by:

$$
\chi_{\mathbf{y}}(\mathbf{x}) := (-1)^{\mathbf{x}\cdot\mathbf{y}}
$$

The fact that $\chi_{\mathbf{y}}$ is a character follows from direct
calculation:

$$
\begin{align*}
\chi_{\mathbf{y}}(\mathbf{x}\oplus\mathbf{x}') &= (-1)^{(\mathbf{x}\oplus\mathbf{x'})\cdot\mathbf{y}} \\
&= (-1)^{\mathbf{x}\cdot\mathbf{y}}\cdot(-1)^{\mathbf{x}'\cdot\mathbf{y}} \\
&= \chi_\mathbf{y}(\mathbf{x})\cdot\chi_\mathbf{y}(\mathbf{x}')
\end{align*}
$$

By claim [Eigenvectors From Characters](#clm:eig-from-char), for all
$\mathbf{x}\in B_N$, the vector

$$
\sum_{\mathbf{x}'}\chi_\mathbf{y}(\mathbf{x}')|\mathbf{x}'\rangle
$$

is an eigenvector of $L_\mathbf{x}$ with eigenvalue
$\chi_\mathbf{y}(\mathbf{x})$. 
We can use these eigenvectors to construct a unitary change of basis that 
simultaneously diagonalizes all of the shift operators:

{: #clm:unitary-diag-old }

> **Claim (Unitary Diagonalization).** Let $U: \mathbb{C}[B_N]\rightarrow\mathbb{C}[B_N]$ be the linear transformation defined by:
>
> $$
> U|\mathbf{y}\rangle = \frac{1}{\sqrt{|B_N|}}\sum_{\mathbf{x}\in B_N}\chi_\mathbf{y}(\mathbf{x})|\mathbf{x}\rangle
> $$
> 
> Then, $U$ is a unitary transformation that diagonalizes the shift operator $L_{\mathbf{x}'}$ 
> for all $\mathbf{x}'\in B_N$.

To prove the claim we will use the following lemma.

> **Lemma.** For all $\mathbf{x}\in B_N$, the shift operator $L_\mathbf{x}$ is a
> real and symmetric transformation.

_Proof of lemma._ Let $\mathbf{x}$ be an element of $B_N$. The fact that $L_\mathbf{x}$ is real
follows imemdiately from the definition of the shift operator.

To prove that $L_\mathbf{x}$ is symmetric we must show that for all $\mathbf{x}',\mathbf{x}''\in B_N$:

$$
\langle \mathbf{x}' | L_\mathbf{x} |\mathbf{x}''\rangle = 
\langle \mathbf{x}'' | L_\mathbf{x} |\mathbf{x}'\rangle
$$

By the definition of the shift operator:

$$
\langle \mathbf{x}' | L_\mathbf{x} |\mathbf{x}''\rangle = 
\langle \mathbf{x}' |\mathbf{x} \oplus \mathbf{x}''\rangle
$$

Therefore:

$$
\langle \mathbf{x}' | L_\mathbf{x} |\mathbf{x}''\rangle =
\begin{cases}
1 & \mathbf{x}' = \mathbf{x} \oplus \mathbf{x}'' \\
0 & \mathrm{else}
\end{cases}
$$

Now, note that the condition $\mathbf{x}' = \mathbf{x} \oplus \mathbf{x}''$ is equivalent
to $\mathbf{x}'\oplus\mathbf{x}'' = \mathbf{x}$ which is symmetric in $\mathbf{x}'$ and $\mathbf{x}''$ which proves the lemma.

_q.e.d_

We can now prove the claim.

_Proof of claim._ By claim [Eigenvectors From Characters](#clm:eig-from-char), for all $\mathbf{x}\in B_N$ 
$U|\mathbf{y}\rangle$ is an eigenvector of $L_\mathbf{x}$ with eigenvalue $\chi_\mathbf{y}(\mathbf{x})$.

Furthermore, since $\|\chi_\mathbf{y}(\mathbf{x})\|=1$ for all $\mathbf{x},\mathbf{y}\in B_N$ it is 
easy to see that the vectors $U|\mathbf{y}\rangle$ all have norm $1$.

Therefore, it suffices to show that $U|\mathbf{y}\rangle$ and $U|\mathbf{y}'\rangle$
are orthogonal for $\mathbf{y}\neq\mathbf{y}'\in B_N$. 

If $\mathbf{y}\neq\mathbf{y}'$, there must be some index
$i$ such that $y_i \neq y'_i$. Let $\mathbf{x}\in B_N$ be defined by $x_i=1$ and $x_j=0$ for $j\neq i$.
It is easy to see that

$$
\chi_\mathbf{y}(\mathbf{x}) \neq \chi_{\mathbf{y}'}(\mathbf{x})
$$

This means that $U|\mathbf{y}\rangle$ and $U|\mathbf{y}'\rangle$ are eigenvectors
of $L_\mathbf{x}$ with different eigenvalues. By the lemma, $L_\mathbf{x}$ is a real symmetric
transformation which means that eigenvectors with distinct eigenvalues are orthogonal.

_q.e.d_

We can describe $U$ more explicitly by plugging in the definition of $\chi_\mathbf{y}(\mathbf{x})$:

$$
U|\mathbf{y}\rangle = \sum_{\mathbf{x}\in B_N} (-1)^{\mathbf{x}\cdot\mathbf{y}}|\mathbf{x}\rangle
$$

From this it is clear that, in addition to being unitary, $U$ is also real and symmetric which means that:

$$
U^{-1} = U^* = U
$$

In other words, the inverse of $U$ is given by:

$$
U^{-1}|\mathbf{x}\rangle = \sum_{\mathbf{y}\in B_N}\chi_\mathbf{y}(\mathbf{x})|\mathbf{y}\rangle
$$

We will define the _Quantum Fourier Transform_ to be the inverse of $U$:

> **Definition (Quantum Fourier Transform of $B_N$).** The Quantum Fourier Transform of $B_N$, denoted 
> $\mathrm{QFT}$, is the transform of $\mathbb{C}[B_N]$ defined by:
>
> $$
> \mathrm{QFT}(|\mathbf{x}\rangle) = \sum_{\mathbf{y}\in B_N}(-1)^{\mathbf{x}\cdot\mathbf{y}}|\mathbf{y}\rangle
> $$

# The Quantum Fourier Transform

Recall from section XXX that our strategy for solving the hidden subgroup problem
is to find a unitary transformation that diagonalizes all of the shift operators $L_g$ for
$g\in G$. This transform is commonly called the
[Quantum Fourier Transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) (QFT).

## Shift Operators

PUT THIS IN THE HSG OVERVIEW

For a group $G$, we will use $\mathbb{C}[G]$ to denote the vector space with one basis
vector for each element of $G$. To be consistent with quantum mechanics notation,
we will denote the basis vector corresponding to $g$ by $|g\rangle$.
Concretely, elements of $\mathbb{C}[G]$ are of the form

$$
\sum_{g\in G}\alpha_g |g\rangle
$$

for complex coefficients $\alpha_g\in\mathbb{C}$.

Let $h\in G$ be an element of $G$. We will define the _Shift Operator_ $L_h$ to
be the following linear transformation of $\mathbb{C}[G]$ defined by:

$$
\begin{align*}
L_h : \mathbb{C}[G] &\rightarrow \mathbb{C}[G] \\
|g\rangle &\mapsto |hg\rangle
\end{align*}
$$

Note that it is sufficient to define $L_h$ on vectors of the form $|g\rangle$ since
they form a basis for $\mathbb{C}[B]$.

As we noted in section XXXX, our strategy for solving the hidden subgroup problem
is to find a unitary transformation of $\

## Characters

As a first step in diagonalizing the shift operators, we'll analyze their eigenvalues.

Suppose that $|\varphi\rangle\in\mathbb{C}[G]$ is an eigenvector of $L_g$ for all $g\in G$.
This means that for all $g\in G$ there is a complex number $\lambda(g)\in\mathbb{C}$ such that

$$
L_g(|\varphi\rangle) = \lambda(g)|\varphi\rangle
$$

We can package these eigenvalues into a function:

$$
\lambda: G \rightarrow \mathbb{C}
$$

What can we say about the function $\lambda$?
Note that since $G$ is associative, for any $g,h\in G$:

$$
\begin{equation}\label{eq:shift-composition}
L_g \circ L_h = L_{gh}
\end{equation}
$$

Applying the left hand side of \ref{eq:shift-composition} to $|\varphi\rangle$
gives:

$$
\begin{align*}
L_g \circ L_h|\varphi\rangle &= L_g(\lambda(h)|\varphi\rangle) \\
&= \lambda(g)\cdot\lambda(h)|\varphi\rangle
\end{align*}
$$

Applying the right hand side of \ref{eq:shift-composition} to $|\varphi\rangle$
gives:

$$
L_{gh}|\varphi\rangle = \lambda(gh)|\varphi\rangle
$$

Together this implies that for all $g,h\in G$:

$$
\lambda(gh) = \lambda(g)\cdot\lambda(h)
$$

Functions satisfying this type of multiplicative property are called
[characters](<https://en.wikipedia.org/wiki/Character_(mathematics)>). More
precisely:

> **Definition (Character)** Let $G$ be a group. A _character_ on $G$ is a
> complex function
>
> $$
> \chi: G \rightarrow \mathbb{C}^\times
> $$
>
> satisfying
>
> $$
> \chi(gh) = \chi(g)\chi(h)
> $$
>
> for all $g,h\in G$.

We can restate the claim above as saying that the eigenvalue function $\lambda$
must be a character on $G$.

We'll denote the set of characters on $G$ by $\hat{G}$. The set $\hat{G}$ has a group
structure where the product of characters $\chi_1,\chi_2\in\hat{G}$ is defined by:

$$
(\chi_1\chi_2)(g) := \chi_1(g)\cdot\chi_2(g)
$$

Here are some basic properties of characters that will be useful below:

{: #clm:character-properties }

> **Claim (Character Properties).** Let $G$ be a finite group and let $N=|G|$ denote the order of $G$.
> Let $\chi$ be a character on $G$. Then:
>
> 1. $\chi(1) = 1$
> 2. $\chi(g)^N = 1$ for all $g\in G$.
> 3. $\chi(g)^{-1} = \bar{\chi} for all $g\in G$.

_Proof._

1. By the definition of characters, $\chi(1)\neq 0$ and:

    $$
    \chi(1)\cdot\chi(1) = \chi(1\cdot 1) = \chi(1)
    $$

    Therefore, $\chi(1) = 1$.

2. By [Lagrange's Theorem](https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(group_theory)),
 $g^N = 1$. Therefore:

    $$
    \chi(g)^N = \chi(g^N) = \chi(1) = 1
    $$

3. By the previous item, $\chi(g)$ is a root of unity.

_q.e.d_

Since this post is focused on abelian groups, it will be helpful to have an explicit
description of characters in that setting. 

Recall that by the
[Fundamental Theorem of Abelian Groups](https://en.wikipedia.org/wiki/Finitely_generated_abelian_group#Primary_decomposition),
any abelian group is a product of cyclic groups. We'll start by describing the characters on
cyclic groups and then show how characters on a product can be understood in terms of the
constituent factors.

{: #clm:cyclic-character-group }

> **Claim (Cyclic Character Group).** Let $N$ be an integer and let $\omega_N\in\mathbb{C}$ be the primitive
> $N$-th root of unity defined by $\omega_N = e^{2\pi i/N}$. For each integer $0 \leq k < N$,
> let $\chi_k: \mathbb{Z}/N \rightarrow \mathbb{C}^\times$ be the function defined by:
>
> $$
> \chi_k(l) = \omega_N^{kl}
> $$
>
>Then:
>
> 1. For all $0 \leq k < N$ the function $\chi_k$ is a character on $\mathbb{Z}/N$
> 1. The following function is a group isomorphism:
>
> $$
> \begin{align*}
> F: \mathbb{Z}/N &\rightarrow \widehat{\mathbb{Z}/N} \\
> k &\mapsto \chi_k
> \end{align*}
> $$

_Proof._ The fact that $\chi_k$ is a character follows directly by the definition of a character.

To see that $F$ is a homomorphism, we must show that $\chi_{k_1 + k_2} = \chi_{k_1}\cdot\chi_{k_2}$.
Let $k_1,k_2\in\mathbb{Z}/N$ be elements of $\mathbb{Z}/N$.
Then for all $l\in\mathbb{Z}/N$:

$$
\begin{align*}
\chi_{k_1 + k_2}(l) &= \omega_N^{(k_1 + k_2)l} \\
&= \omega_N^{k_1 l}\cdot \omega_N^{k_2 l} \\
&= \chi_{k_1}(l)\cdot\chi_{k_2}(l)
\end{align*}
$$

To see that $F$ is injective, note that if $k\in\mathbb{Z}/N$ is in the kernel of $F$
then $\chi_k(l) = \omega_N^{kl} = 1$ for all $l\in\mathbb{Z}/N$. But this implies that $k=0$.

To see that $F$ is surjective, let $\chi\in\widehat{Z/N}$ be a character. By
[Character Properties](#clm:character-properties), $\chi(1)^N=1$. Therefore, there exists some
$k\in\mathbb{Z}/N$ such that $\chi(1) = \omega_N^k$. It is easy to see that $\chi = F(k)$.

_q.e.d_

Now suppose that the group $G$ is a product:

$$
G = \prod_{l=1}^L G_l
$$

If $\chi_l\in\hat{G}_l$ is a character on $G_l$ then
we can define a character $P(\chi_1,\dots,\chi_L)$ on $G$ by:

$$
\begin{align*}
P(\chi_1,\dots,\chi_L): G &\rightarrow \mathbb{C}^\times \\
(g_1,\dots,g_l) &\mapsto \prod_{l=1}^L\chi_l(g_l)
\end{align*}
$$

This defines a function:

$$
P: \prod_{l=1}^L \hat{G_l} \rightarrow \hat{G}
$$

The following claim says that all characters on $G$ are of this form.

{: #clm:abelian-character-group}

> **Claim (Product Character Group).** Let the finite group $G$ be a product:
>
> $$
> G = \prod_{l=1}^L G_l
> $$
>
> Then the function
>
> $$
> P: \prod_{l=1}^L \hat{G_l} \rightarrow \hat{G}
> $$
>
> defined above is an isomorphism of groups.

_Proof_. It is easy to see that $P$ is a homomorphism. To see that $P$ is injective,
suppose that $(\chi_1,\dots,\chi_L)$ is in the kernel of $P$. That implies that for all
$(g_1,\dots,g_L)\in G$:

$$
P(\chi_1,\dots,\chi_L)(g_1,\dots,g_L) = \prod_{l=1}^L\chi_l(g_l) = 1
$$

In particular, if we choose $g_l$ to be the identity for all $l > 1$ then we get that:

$$
\prod_{l=1}^L\chi_l(g_l) = \chi_1(g_1) = 1
$$

for all $g_1\in G_1$. This implies that $\chi_1 = 1$. The same argument shows that
$\chi_l = 1$ for all $l$ which means that the kernel of $P$ is trivial.

To show that $P$ is surjective, let $\chi$ be a character on $G$.
The restriction of $\chi$ to $G_l$ is a character on $G_l$ which we will denote by $\chi_{G_l}$.
It is easy to see that $P(\chi_{G_1},\dots,\chi_{G_L}) = \chi$.

_q.e.d_

The next claim follows immediately from the previous two claims and the
[Fundamental Theorem of Abelian Groups](https://en.wikipedia.org/wiki/Finitely_generated_abelian_group#Primary_decomposition).

{: #clm:abelian-character-group}

> **Claim (Abelian Character Group).** For any finite abelian group $G$, $G\cong\hat{G}$.

## Eigenvectors

In the previous section we saw that the eigenvalues of $L_g$ are given by characters
of $G$. Interestingly, given a character $\chi\in\hat{G}$, it is easy to construct
a simultaneous eigenvector for $L_g$ whose eigenvalues are given by $\chi$:

{: #clm:eig-from-char }

> **Claim (Eigenvectors From Characters).** Let $G$ be a finite group and let
> $\chi: G \rightarrow \mathbb{C}$ be a character on $G$. Define
> $|\varphi\rangle\in\mathbb{C}[G]$ by:
>
> $$
> |\varphi\rangle = \sum_{g\in G}\chi^*(g)|g\rangle
> $$
>
> Then, for all $g\in G$, $|\varphi\rangle$ is an eigenvector of
> $L_g$ with eigenvalue $\chi(g)$.

_Proof._ Recall that by [Character Properties](#clm:character-properties), 
$\chi^*(g)=\chi^{-1}(g)$ for all $g\in G$. The claim follows by direct calculation:

$$
\begin{align*}
L_g|\varphi\rangle &= L_g\sum_{h}\chi^*(h)|h\rangle \\
&= \sum_{h\inG}\chi^*(h)|gh\rangle \\
&= \sum_{h\in G}\chi^*(g^{-1}h)|g g^{-1}h\rangle \\
&= \sum_{h\in G}\chi^*(g^{-1})\chi(h)|h\rangle \\
&= \chi(g)\cdot\sum_{h\in G}\chi(h)|h\rangle \\
&= \chi(g)|\varphi\rangle
\end{align*}
$$

_q.e.d_

We can use this result to simultaneously diagonalize the shift operators $L_g$.

To facilitate notation, for each $g\in G$ we will define $D_g$ to be the diagonal transformation
of $\mathbb{C}[\hat{G}]$ defined by:

$$
\begin{align*}
D_g: \mathbb{C}[\hat{G}] &\rightarrow \mathbb{C}[\hat{G}] \\
|\chi\rangle &\mapsto \chi(g)|\chi\rangle
\end{align*}
$$

{: #clm:unitary-diag }
> **Claim (Diagonalization).** Let $U$ be the linear transformation defined by:
>
> $$
> \begin{align*}
> U : \mathbb{C}[\hat{G}] &\rightarrow \mathbb{C}[G] \\
> |\chi\rangle &\mapsto \frac{1}{\sqrt{|G|}}\sum_{g\in G}\bar{\chi(g)}|g\rangle
> \end{align*}
> $$
>
> Then, $U$ is unitary and for all $g\in G$:
>
> $$
> L_g = U D_g U^*
> $$

In other words, $U$ is a [unitary transformation](https://en.wikipedia.org/wiki/Unitary_transformation) that simultaneously diagonalizes
$L_g$ for all $g\in G$.

_Proof._ By [Abelian Character Group](#clm:abelian-character-group), $\mathbb{C}[\hat{G}]$
and $\mathbb{C}[G]$ have the same dimension. Therefore, it suffices to show that $U$
maps basis vectors $|\chi\rangle\in\mathbb{C}[G]$ to orthogonal unit eigenvectors of $L_g$.

We'll start by showing orthogonality. Let $\chi_1\neq\chi_2\in\hat{G}$ be different characters on $G$.
Since $\chi_1\neq\chi_2$, there must be some $g\in G$ such that $\chi_1(g)\neq\chi_2(g)$.
By claim [Eigenvectors From Characters](#clm:eig-from-char), this implies that $U|\chi_1\rangle$
and $U|\chi_2\rangle$ are eigenvectors of $L_g$ with different eigenvalues. Since $L_g$ is a permutation
transformation, [it follows](https://en.wikipedia.org/wiki/Permutation_matrix#Linear-algebraic_properties)
that is [normal](https://en.wikipedia.org/wiki/Normal_matrix). Eigenvectors of a normal transformation
with distinct eigenvalues are orthogonal which implies that $U|\chi_1\rangle$ and $U|\chi_s\rangle$
are orthogonal.

It remains to show that $U|\chi\rangle$ has unit length for all $\chi\in\hat{G}$.
Recall that by [Character Properties](#clm:character-properties), $\bar{\chi}(g)=\chi^{-1}(g)$ for all
$g\in G$ which implies that $\|\chi^*(g)\|^2 = 1$. Therefore:

$$
||U|\chi\rangle||^2 = \frac{1}{G}\sum_{g\in G}|\chi^*(g)|^2 = 1
$$

_q.e.d_

We will define the _Quantum Fourier Transform_ of $G$ to be complex conjugate of $U$:

{: #defn:quantum-fourier-transform }

> **Definition (Quantum Fourier Transform).** Let $G$ be a finite abelian group.
> The Quantum Fourier Transform (QFT) on $G$ is defined by:
>
> $$
> \begin{align*}
> \mathrm{QFT}_G : \mathbb{C}[G] &\rightarrow \mathbb{C}[\hat{G}] \\
> |g\rangle &\mapsto \frac{1}{\sqrt{|G|}}\sum_{\chi\in \hat{G}}\chi(g)|\chi\rangle
> \end{align*}
> $$

The following claim is a direct consequence of [Diagonalization](#clm:unitary-diag):

{: #clm:qft-diagonalization }

> **Claim (QFT Diagonalization).** Let $G$ be a finite abelian group.
> Then $\mathrm{QFT}_G$ is unitary and for all $g\in G$:
>
> $$
> L_g = \mathrm{QFT}_G^* \circ D_g \circ \mathrm{QFT}_G
> $$

_Proof._ This follows immediately from [Diagonalization](#clm:unitary-diag) and the fact that
$\mathrm{QFT} = U^*$

_q.e.d_

It is common to construct groups as products of smaller groups. Suppose that $G$ and $H$
are finite abelian groups and that we know $\mathrm{QFT}_G$ and $\mathrm{QFT}_H$. What can we say about
$\mathrm{QFT}_{G\times H}$, the QFT on the product $G\times H$?

First of all, note that $\mathbb{C}[G\times H]$ is isomorphic to $\mathbb{C}[G]\otimes\mathbb{C}[H]$
with the isomorphism given by:

$$
\begin{align*}
\mathbb{C}[G]\otimes\mathbb{C}[H] &\rightarrow \mathbb{C}[G\times H] \\
|g\rangle \otimes |h\rangle &\mapsto |(g,h)\rangle
\end{align*}
$$

We will therefore slightly abuse notation and identify $\mathbb{C}[G\times H]$ with
$\mathbb{C}[G]\otimes\mathbb{C}[H]$. We can use this identification to compare
$\mathrm{QFT}\_{G\times H}$ to $\mathrm{QFT}\_{G}$ and $\mathrm{QFT}\_{H}$

{: #clm:qft-product }

> **Claim (QFT Product).** Let $G$ and $H be finite abelian groups. Then:
>
> $$
> \mathrm{QFT}_{G\times H} = \mathrm{QFT}_G \otimes \mathrm{QFT}_H
> $$

_Proof._ First, let's see how the characters on $G\times $H relate to the characters on
$G$ and $H$. Suppose that $\chi\in\hat{G}$ is a character on $G$ and $\lambda\in\hat{H}$
is a character on $H$. It is easy to see that the function $(\chi,\lambda)$ defined by: 

$$
\begin{align*}
(\chi,\lambda): G\times H &\rightarrow \mathbb{C}^\times \\
(g, h) &\mapsto \chi(g)\lambda(h)
\end{align*}
$$

is a character on $G\times H$. This defines a group homomorphism:

$$
\hat{G} \times \hat{H} \rightarrow \widehat{G\times H}
$$

It is easy to see that it is injective. Furthermore, 
by [Abelian Character Group](#clm:abelian-character-group) the groups on both sides have
order $|G|\cdot|H|$. Therefore this correspondence is an isomorphism. As a consequence we have:

$$
\mathbb{C}[\widehat{G\times H}] \cong \mathbb{C}[\hat{G}]\otimes\mathbb{C}[\hat{H}]
$$

Now, let $g\in G$ be an element of $G$ and $h\in H$ be an element of $H$. By the above identification 
of $\widehat{G\times H}$ and $\hat{G}\times\hat{H}$ together with the definition
of $\mathrm{QFT}_{G\times H}$:

$$
\begin{align*}
\mathrm{QFT}_{G\times H}|(g,h)\rangle &= \frac{1}{\sqrt{|G\times H|}}\sum_{\chi\in \widehat{G\times H}}\chi((g,h))|\chi\rangle \\
&= \frac{1}{\sqrt{|G|\cdot| H|}}\sum_{\chi\in \hat{G},\,\lambda\in\hat{H}}\chi(g)\lambda(h)|\chi\rangle|\lambda\rangle \\
&= \left(\frac{1}{\sqrt{|G|}}\sum_{\chi\in \hat{G}}\chi(g)|\chi\rangle\right) \otimes \left(\frac{1}{\sqrt{|H|}}\sum_{\lambda\in \hat{H}}\lambda(h)|\lambda\rangle\right) \\
&= \mathrm{QFT}_G(g) \otimes \mathrm{QFT}_H(h) 
\end{align*}
$$

_q.e.d_


## Coset States

We'll now see how the quantum fourier transform can be used to solve the hidden subgroup problem.
Recall from section XXX that the first part of the _standard method_ produces a coset state:

$$
|gH\rangle = \frac{1}{\sqrt{|H|}}\sum_{h\in H}|gh\rangle
$$

where $H$ is the hidden subgroup and $g$ is an unknown element of $G$. We'll use the QFT
to extract information about $H$ from a coset state $|gH\rangle$.

If $\chi\in\hat{G}$ is a character on $G$ and $H\subset G$ is a subgroup of $G$
then we will use the notation $\chi_H$ to denote the restriction of $\chi$ to $H$.

Let's see what happens if we apply the QFT to $\|H\rangle$:

{: #clm:qft-subgroup-state }

> **Claim (QFT Subgroup State).** Let $G$ be a finite abelian group and let $H\subset G$ be
> a subgroup. Then:
>
> $$
> \mathrm{QFT}(|H\rangle) = \sqrt{\frac{|H|}{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}|\chi\rangle
> $$

In other words, applying the QFT to $|H\rangle$ gives a sum over the characters on $G$
whose restriction to $H$ is trivial.

_Proof._ First, note that for all $h\in H$ the state $\|H\rangle$ is invariant under
a shift by $h$:

$$
\begin{equation}\label{eq:subgroup-state-shift}
L_h|H\rangle = |H\rangle
\end{equation}
$$

By [QFT Diagonalization](#clm:qft-diagonalization), $\mathrm{QFT}$ is unitary and:

$$
L_h = \mathrm{QFT}^* \circ D_h \circ \mathrm{QFT}
$$

Applying this to \ref{eq:subgroup-state-shift} gives and multiplying both sides
by $\mathrm{QFT}$ gives:

$$
\mathrm{QFT}|H\rangle = D_h\mathrm{QFT}|H\rangle 
$$

Taking the inner product of both sides with a character $\chi\in\hat{G}$ and
using the definition of $D_h$:

$$
\begin{align*}
\langle\chi|\mathrm{QFT}|H\rangle &= \langle\chi|D_h\mathrm{QFT}|H\rangle \\
&= \chi(h)\langle\chi|\mathrm{QFT}|H\rangle
\end{align*}
$$

Therefore, either $\langle\chi\|\mathrm{QFT}\|H\rangle=0$ or $\chi(h)=1$ for all $h\in H$.

Suppose that $\chi_H=1$. Then:

$$
\begin{align*}
\langle\chi|\mathrm{QFT}|H\rangle &= \frac{1}{\sqrt{|G|}\sqrt{|H|}}\sum_{h\in H}\chi(h)|\chi\rangle \\
&= \frac{|H|}{\sqrt{|G|}\sqrt{|H|}}\sum_{h\in H}|\chi\rangle \\
&= \sqrt{\frac{|H|}{|G|}}|\chi\rangle
\end{align*}
$$

_q.e.d_

We can use this claim together with another application of [QFT Diagonalization](#clm:qft-diagonalization) to determine what
happens when we apply the QFT to a coset state:

{: #clm:qft-coset-state }
> **Claim (QFT Coset State).** Let $G$ be a finite abelian group and let $H\subset G$ be
> a subgroup and $g\in G$ and element of $G$. Then:
>
> $$
> \mathrm{QFT}(|gH\rangle) = \sqrt{\frac{|H|}{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}\chi(g)|\chi\rangle
> $$

_Proof._ We can rewrite the coset state as:

$$
|gH\rangle = L_g|H\rangle
$$

By [QFT Diagonalization](#clm:qft-diagonalization), $L_g = \mathrm{QFT}^*\circ D_g\circ \mathrm{QFT}$ and so:

$$
\mathrm{QFT}|gH\rangle = (D_g\circ\mathrm{QFT})|H\rangle
$$

Applying [QFT Subgroup State](#clm:qft-subgroup-state) gives:

$$
\begin{align*}
\mathrm{QFT}|gH\rangle &= D_g \frac{\sqrt{|H|}}{\sqrt{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}|\chi\rangle \\
&= \frac{\sqrt{|H|}}{\sqrt{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}D_g|\chi\rangle \\
&= \frac{\sqrt{|H|}}{\sqrt{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}\chi(g)|\chi\rangle 
\end{align*}
$$

_q.e.d_

This gives us a strategy for extracting information about $H$ from an arbitrary coset state $|gH\rangle$.
According to [QFT Coset State](#clm:qft-coset-state), if we apply the QFT to $|gH\rangle$ and measure in
the standard basis of $\mathbb{C}[\hat{G}]$ then we are guaranteed to measure a character $\chi$
whose restriction to $H$ is trivial. As we will see in the following sections, this essentially imposes
a linear constraint on the generators of $H$. Repeating the process $\mathcal{O}(\log\|H\|)$ times will determine
the generators of $H$ with high probability.

# Simon's Algorithm

In section XXX we introduced Simon's problem and saw that it is a special case of
the hidden subgroup problem with $G = (\mathbb{Z}/2)^N$ and function $f:G\rightarrow A$ 
that hides a subgroup $H$ of order $2$.

In this section we'll apply the theory from section [The Quantum Fourier Transform](#the-quantum-fourier-transform)
to this group and recover Simon's original solution to the problem.

We'll represent elements of $(\mathbb{Z}/2)^N$ as bit-strings 

$$
\mathbf{x} = x_1,\dots,x_n
$$

where $x_i\in\{0,1\}$. The group operation on $(\mathbb{Z}/2)^N$ is bitwise addition modulo $2$
which we will denote by $\oplus$.

## Characters

In order to construct the Quantum Fourier Transform we need to
find the characters on $(\mathbb{Z}/2)^N$.

To facilitate notation, we will introduce the bitwise dot product on elements of
$(\mathbb{Z}/2)^N$:

$$
\mathbf{x} \cdot \mathbf{y} := \sum_{i=1}^N x_iy_i
$$

Given a bit-string $\mathbf{y}\in(\mathbb{Z}/2)^N$, we'll define the character $\chi_\mathbf{y}$
to be:

$$
\begin{align*}
\chi_\mathbf{y}: (\mathbb{Z}/2)^N &\rightarrow \mathbb{C}^\times \\
\mathbf{x} &\mapsto (-1)^{\mathbf{x}\cdot\mathbf{y}}
\end{align*}
$$

To check that $\chi_\mathbf{y}$ is a character, let $\mathbf{x},\mathbf{x}'\in (\mathbb{Z}/2)^N$
be to bit-strings. Then indeed:

$$
\begin{align*}
\chi_{\mathbf{y}}(\mathbf{x}\oplus\mathbf{x}') &= (-1)^{(\mathbf{x}\oplus\mathbf{x}')\cdot\mathbf{y}} \\
&= (-1)^{(\mathbf{x}\cdot\mathbf{y})\oplus(\mathbf{x}'\cdot\mathbf{y})} \\
&= (-1)^{(\mathbf{x}\cdot\mathbf{y})}\cdot (-1)^{(\mathbf{x}'\cdot\mathbf{y})} \\
&= \chi_\mathbf{y}(\mathbf{x})\cdot\chi_\mathbf{y}(\mathbf{x}')
\end{align*}
$$

By [Abelian Character Group](#clm:abelian-character-group), there are $2^N$ characters on
$(\mathbb{Z}/2)^N$ which means that all of the characters on $(\mathbb{Z}/2)^N$ are of the form
$\chi_\mathbf{y}$ for some $\mathbf{y}\in (\mathbb{Z}/2)^N$.

By [definition](#defn:quantum-fourier-transform), this means that the QFT on this group is given by:

$$
\mathrm{QFT}(|\mathbf{x}\rangle) = 
\frac{1}{2^{N/2}}\sum_{\mathbf{y}\in(\mathbb{Z}/2)^N}(-1)^{\mathbf{x}\cdot\mathbf{y}}|\mathbf{y}\rangle
$$

Note that in this sum, we are representing the character basis elements $\|\chi_\mathbf{y}\rangle$ by
the bit-string $\|\mathbf{y}\rangle$.

## The Standard Method

We now use the QFT to use the standard method from section XXX to solve Simon's problem.

First, recall in this case the hidden subgroup $H$ is guaranteed to be of order $2$.
So we'll write it as:

$$
H = \{\mathbf{0}, \mathbf{s} \}
$$

for some hidden element $\mathbf{s}\in(\mathbb{Z}/2)^N$ that we are trying to find.

As usual, the first step of the standard method is to construct the state:

$$
\frac{1}{2^{N/2}}\sum_{\mathbf{x}\in(\mathbb{Z}/2)^N}|\mathbf{x}\rangle|0\rangle
$$

Next we apply $f$ to obtain:

$$
\frac{1}{2^{N/2}}\sum_{\mathbf{x}\in(\mathbb{Z}/2)^N}|\mathbf{x}\rangle|f(\mathbf{x})\rangle
$$

Then we measure the auxiliary register and are left with a coset state:

$$
|\mathbf{x} \oplus H\rangle = \frac{1}{\sqrt{2}}(|\mathbf{x}\rangle + |\mathbf{x}\oplus\mathbf{s}\rangle)
$$

for some unknown $mathbf{x}\in(\mathbb{Z}/2)^N$.

By [QFT Coset State](#clm:qft-coset-state), we expect that applying the QFT to 
$|\mathbf{x} \oplus H\rangle$ should remove confounding effect of $\mathbf{x}$ and reveal information about $\mathbf{s}$.

Indeed, using the explicit form for the QFT in this case from the previous section:

$$
\begin{align*}
\mathrm{QFT}($|\mathbf{x} \oplus H\rangle) &= 
\frac{1}{2^{N/2}}\sum_{\mathbf{y}\in(\mathbb{Z}/2)^N}((-1)^{\mathbf{x}\cdot\mathbf{y}} + (-1)^{(\mathbf{x}\oplus\mathbf{s})\cdot\mathbf{y}}|\mathbf{y}\rangle \\
&= \frac{1}{2^{N/2}}\sum_{\mathbf{y}\in(\mathbb{Z}/2)^N}((-1)^{\mathbf{x}\cdot\mathbf{y}}(1 + (-1)^{\mathbf{s}\cdot\mathbf{y}})|\mathbf{y}\rangle
\end{align*}
$$

Note that the coefficient of $|\chi_\mathbf{y}\rangle$ is either equal to $0$ or $2$:

$$
1 + (-1)^{\mathbf{s}\cdot\mathbf{y}} = \begin{cases}
2 & \mathbf{s}\cdot\mathbf{y} = 0 \\
0 & \mathrm{else}
$$

Therefore:

$$
\mathrm{QFT}($|\mathbf{x} \oplus H\rangle) =
\frac{1}{2^{(N-1)/2}}\sum_{\mathbf{y}\in(\mathbb{Z}/2)^N,\,\mathbf{s}\cdot\mathbf{y}=0}(-1)^{\mathbf{x}\cdot\mathbf{y}}|\mathbf{y}\rangle
$$

as predicted by claim [QFT Coset State](#clm:qft-coset-state).

This means that if we measure $\mathrm{QFT}($|\mathbf{x} \oplus H\rangle)$ in the standard basis
we get an element $\mathbf{y}\in(\mathbb{Z}/2)^N$ satisfying

$$
\mathbf{s}\cdot\mathbf{y} = \sum_{i=1}^N s_iy_i = 0 \mathrm{(mod 2)}$.
$$

In other words, after measuring $\mathrm{QFT}(|\mathbf{x} \oplus H\rangle)$ we will get one linear
constraint on the hidden vector $\mathbf{s}$. If we repeat the process $\mathcal{O}(N)$ times, we have a high
chance of finding $N$ independent linear constraints on the length $N$ vector $\mathbf{s}$ which
is sufficient to recover $\mathbf{s}$ and solve Simon's problem.

In conclusion, we can use the QFT to solve Simon's problem in only $\mathcal{O}(N)$ invocations of $f$.
In the next section we'll show how to efficiently implement the QFT in this case.

## QFT Implementation

In this section we'll show how to efficiently implement the QFT on $(\mathbb{Z}/2)^N$.
By _efficiently_ we mean that the quantum circuit for the QFT should consist of $\mathcal{O}(N)$ gates,
where each gate operates on no more than two bits.

Recall that the QFT on $(\mathbb{Z}/2)^N$ is given by:

$$
\mathrm{QFT}_{(\mathbb{Z}/2)^N}(|\mathbf{x}\rangle) = 
\frac{1}{2^{N/2}}\sum_{\mathbf{y}\in(\mathbb{Z}/2)^N}(-1)^{\mathbf{x}\cdot\mathbf{y}}|\mathbf{y}\rangle
$$

Since there are $2^N$ terms in the sum, a naive implementation that computes the QFT term by term would
be too slow.

Instead, we will implement the QFT using [Hadamard Gates](https://en.wikipedia.org/wiki/Quantum_logic_gate#Hadamard_gate).
Recall that a Hadamard gate $H$ on a single qubit is defined by:

$$
\begin{align*}
H|0\rangle &= \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \\
H|1\rangle &= \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)
\end{align*}
$$

We can write this more succinctly as:

$$
H|x\rangle = \frac{1}{\sqrt{2}}(|0\rangle + (-1)^x|1\rangle)
$$

where $x\in\mathbb{Z}/2$. But note that this is equal to the QFT on $\mathbb{Z}/2$.
In other words:

$$
\mathrm{QFT}_{\mathbb{Z}/2} = H
$$


The following claim follows immediately from this observation and [QFT Product](#clm:qft-product):

> **Claim (QFT For Bit-strings).** The QFT on $(\mathbb{Z}/2)^N$ can be expressed as:
>
> $$
> \mathrm{QFT}_{(\mathbb{Z}/2)^N} = H^{\otimes N}
> $$

# Shor's Algorithm

In section XXX we saw that computing [discrete logarithms](https://en.wikipedia.org/wiki/Discrete_logarithm)
can be reduced to solving the hidden subgroup problem on the group 

$$
G=\mathbb{Z}/N\times\mathbb{Z}/N
$$

for an integer $N$.

Similarly to section [Simon's Algorithm](#simons-algorithm), in this section we'll
compute the QFT for $\mathbb{Z}/N \times \mathbb{Z}/N$ and use it to derive Shor's algorithm for the
discrete logarithm. 

## Characters

In order to compute $\mathrm{QFT}_{\mathbb{Z}/N \times \mathbb{Z}/N}$ we need to
describe the characters on $\mathbb{Z}/N \times \mathbb{Z}/N$.

Let $\omega_N\in\mathbb{C}$ be the primitive $N$-th root of unity defined by:

$$
\omega_N = e^{2\pi i/N}
$$

For an integer $k$, let $\chi_k: \mathbb{Z}/N \rightarrow\mathbb{C}^\times$ be the
function defined by:

$$
\chi_k(l) = \omega^{kl}
$$

By [Cyclic Character Group](#clm:cyclic-character-group), the characters on $\mathbb{Z}/N$
are all of the form $\chi_k$ for some $0 \leq k < N$.

By [Abelian Character Group](#clm:abelian-character-group), the characters on 
We'll denote the primitive $N$-th root of unity by $\omega_N$:

$$
\omega_N = e^{2\pi i / N}
$$

Let $\chi\in\widehat{\mathbb{Z}/N}$ be a character on $\mathbb{Z}/N$.
First note that $\chi$ is determined by its value on $1\in\mathbb{Z}/N$ since for
all $0 \leq l < N$:

$$
\chi(l) = \chi(l\cdot 1) = \chi(1)^l
$$

In addition, by [Character Properties](#clm:character-properties):

$$
\chi(1)^N = 1
$$

Therefore, there is some $k\in\mathbb{Z}$ satisfying $0\leq k < N$ such that:

$$
\chi(1) = \omega_N^k
$$

Together this implies that the characters on $\mathbb{Z}/N$ are all of the form:

$$
\chi(l) = \omega_N^{kl}
$$

for some $0\leq k < N$.

By [definition](#defn:quantum-fourier-transform), it follows that the QFT on
$\mathbb{Z}/N$ is given by:

$$
\mathrm{QFT}_{\mathbb{Z}/N}(|l\rangle) = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1}\omega_N^{kl}|k\rangle
$$

## The Standard Method

Similarly to Simon's Algorithm, we can use $\mathrm{QFT}_{\mathbb{Z}/N}$ to apply the [standard method](XXX)
to solve the hidden subgroup problem on $\mathbb{Z}/N$.


In this case the hidden subgroup $H$ will be generated by an element $l\in\mathbb{Z}/N$ such that
$l$ divides $N$. Therefore, determining $H$ is equivalent to finding the generator $l$.
Furthermore, note that:

$$
|H| = N / l
$$

As usual, the first step of the standard method will produce a coset state:

$$
|m + H\rangle
$$

for some unknown $m\in\mathbb{Z}/N$. By theorem [QFT Coset State](#clm:qft-coset-state),
applying $\mathrm{QFT}_{\mathbb{Z}/N}$ to this state will give us:

USE \chi_k to make this more clear. i.e, (\chi_k)_H=1 iff \omega_N^{kl}=1
$$
\mathrm{QFT}_{\mathbb{Z}/N}|m + H\rangle = \frac{1}{l}\sum_{k\in\mathbb{Z}/N,\,\omega_N^{kl}=1}\omega_N^{km}|k\rangle
$$



























