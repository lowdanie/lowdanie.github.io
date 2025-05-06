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

First of all, note that for all $\mathbf{x}\in B_N$, the square of
$L_\mathbf{x}$ is the identity:

$$
\begin{align*}
L_\mathbf{x}^2 &= L_\mathbf{x} \circ L_\mathbf{x} = L_{\mathbf{x}\oplus\mathbf{x}} \\
&= L_\mathbf{0} = \mathrm{Id}
\end{align*}
$$

This implies that the eigenvalues of $L_\mathbf{x}$ are equal to $\pm 1$.

Suppose that $|\mathbf{y}\rangle\in\mathbb{C}[B_N]$ is a vector in the new
basis. We can write $|\mathbf{y}\rangle$ in terms of the standard basis as:

$$
|\mathbf{y}\rangle = \sum_{\mathbf{x}\in B_N} \chi_\mathbf{y}(\mathbf{x})|\mathbf{x}\rangle
$$

for some constants $\chi_\mathbf{y}(\mathbf{x})\in\mathbb{C}$. Since
$|\mathbf{y}\rangle$ is only determined up to scalar, we can assume that
$\chi_\mathbb{y}(\mathbb{0})=1$.

Let $\mathbf{x}$ be an element of $B_N$. Since $L_\mathbf{x}$ is diagonal in the
new basis, there must be some complex scalar
$\lambda_\mathbf{y}(\mathbf{x})=\pm 1$ such that:

$$
L_\mathbf{x}(|\mathbf{y}\rangle) = \lambda_\mathbf{y}(\mathbf{x}) |\mathbf{y}\rangle
$$

By the definition of $L_\mathbf{x}$:

$$
\begin{align*}
L_\mathbf{x}(|\mathbf{y}\rangle) &= L_\mathbf{x}( \sum_{\mathbf{x}'\in B_N} \chi_\mathbf{y}(\mathbf{x}')|\mathbf{x}'\rangle) \\
&=  \sum_{\mathbf{x}'\in B_N} \chi_\mathbf{y}(\mathbf{x}') L_\mathbf{x}(|\mathbf{x}'\rangle) \\
&=  \sum_{\mathbf{x}'\in B_N} \chi_\mathbf{y}(\mathbf{x}')|\mathbf{x}\oplus\mathbf{x}'\rangle
\end{align*}
$$

Together we get that:

$$
\sum_{\mathbf{x}'\in B_N} \chi_\mathbf{y}(\mathbf{x}')|\mathbf{x}\oplus\mathbf{x}'\rangle =
\lambda_\mathbf{y}(\mathbf{x}) \sum_{\mathbf{x}'\in B_N} \chi_\mathbf{y}(\mathbf{x}')|\mathbf{x}'\rangle
$$

Comparing the coefficients in each sum, we see that for each
$\mathbf{x}'\in B_N$:

$$
\chi_\mathbf{y}(\mathbf{x}') = \lambda_\mathbf{y}(\mathbf{x}) \cdot \chi_\mathbf{y}(\mathbf{x}\oplus\mathbf{x}')
$$

Since $\lambda_\mathbf{y}(\mathbf{x})^2 = 1$, we can rearrange this to:

$$
\chi_\mathbf{y}(\mathbf{x}\oplus\mathbf{x}') = \lambda_\mathbf{y}(\mathbf{x}) \cdot \chi_\mathbf{y}(\mathbf{x}')
$$

Plugging in $\mathbf{x}' = \mathbf{0}$ and using our normalization
$\chi_\mathbf{y}(\mathbf{0})=1$ gives us:

$$
\chi_\mathbf{y}(\mathbf{x}) = \lambda_\mathbf{y}(\mathbf{x}) \cdot \chi_\mathbf{y}(\mathbf{0}) =  \lambda_\mathbf{y}(\mathbf{x})
$$

In summary:

$$
\chi_\mathbf{y}(\mathbf{x}\oplus\mathbf{x}') = \chi_\mathbf{y}(\mathbf{x}) \cdot \chi_\mathbf{y}(\mathbf{x}')
$$
