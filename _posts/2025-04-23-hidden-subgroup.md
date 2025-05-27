---
layout: post
title: "The Hidden Subgroup Problem"
date: 2025-04-23
mathjax: true
utterance-issue: 12
---

# Introduction

Two of the most famous [quantum algorithms](https://en.wikipedia.org/wiki/Quantum_algorithm)
 are
[Shor's Algorithms](https://en.wikipedia.org/wiki/Shor%27s_algorithm) for
[integer factorization](https://en.wikipedia.org/wiki/Integer_factorization) and
[discrete logarithms](https://en.wikipedia.org/wiki/Discrete_logarithm).
The significance of these algorithms is that efficient integer factorization breaks 
the [RSA cryptosystem](https://en.wikipedia.org/wiki/RSA_cryptosystem),
 and efficient discrete logarithm computation
breaks the [Diffie-Hellman](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange)
key exchange protocol.

It is natural to wonder what other types of problems that are classically considered
to be hard can be efficiently solved with a quantum computer.

Interestingly, both integer factorization and the discrete logarithm problem 
are special cases of a more general problem called the
[hidden subgroup problem](https://en.wikipedia.org/wiki/Hidden_subgroup_problem) (HSP).
Furthermore, Shor's algorithms can be generalized to an efficient quantum
algorithm for the HSP whenever the group in question is
[commutative](https://en.wikipedia.org/wiki/Abelian_group).

Conversely, [almost all](https://en.wikipedia.org/wiki/Hidden_subgroup_problem#Instances)
known problems with significant quantum speedups are
instances of the HSP for commutative groups.

In this post we'll introduce the hidden subgroup problem and the quantum 
algorithm that efficiently solves it in the commutative case. We'll also apply this
framework to the discrete logarithm problem and
[Simon's Problem](https://en.wikipedia.org/wiki/Simon%27s_problem).

# The Hidden Subgroup Problem

In this section we'll formally define the hidden subgroup problem and then see
how it generalizes some famous problems in quantum computing.

We'll start by defining the notion of a function on a group _hiding_ a subgroup.

{: #defn:hiding-a-subgroup }

> **Definition (Hiding A Subgroup).** Let $G$ be a finite group, $H\subset G$ a
> subgroup of $G$, $A$ a set and
>
> $$
> f: G \rightarrow A
> $$ 
> 
> a function from $G$ to $A$. 
> We say that $f$ _hides_ $H$ if, for all $g_1,g_2\in G$,
> $f(g_1) = f(g_2)$ if and only if $g_1H = g_2H$.

In other words, $f$ hides $H$ if it defines an injective function on the cosets of $H$.

The _hidden subgroup problem_ is to find $H$ given an black box which allows
us to evaluate $f$ on elements of $G$.

{: #defn:hidden-subgroup-problem }

> **Definition (Hidden Subgroup Problem).** Let $G$ be a finite group, $H\subset G$
> a subgroup and 
> $f: G \rightarrow A$ a function that hides $H$.
> The objective of the _hidden subgroup problem_ (HSP) is to find generators for 
> $H$ using evaluations of a black box for $f$.

It's always possible to solve an instance of the hidden subgroup problem 
with $\mathcal{O}(|G|)$ evaluations of $f$. Indeed, we can evaluate $f$ on every element of
$G$ and note that:

$$
H = \{g\in G | f(g) = f(e) \}
$$

where $e\in G$ denotes the identity element in $G$.

The interesting question is whether we can find generators for $H$ more efficiently.

When $G$ is [abelian](https://en.wikipedia.org/wiki/Abelian_group), there is a generalization
of
[Shor's algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm) 
called _the standard method_ which
solves the HSP with only $\mathcal{O}(\log(|G|)$ invocations of
$f$. In contrast, there are many interesting instances of the ableian HSP where the best known
classical algorithm requires $\mathcal{O}(\sqrt{|G|})$ invocations.

Below we'll consider two important instances of the abelian HSG. We'll then describe
the standard method and apply it to each example.

## Simon's Problem

Simon's problem was introduced by
[Daniel Simon](https://epubs.siam.org/doi/10.1137/S0097539796298637) in 1994 as
an example of a problem which classically requires nearly exponential time, but can be
efficiently solved with a quantum computer.

Let $(\mathbb{Z}/2)^N$ be the group of bit-strings of length $N$ and let
$\oplus$ denote the bitwise XOR operation.

For example, if $N=3$ then $110$ and $101$ are elements of $(\mathbb{Z}/2)^3$ and:

$$
110 \oplus 101 = 011
$$

Simon's problem is defined as follow:

> We are given access to a black-box for computing a function
> 
> $$
> f: (\mathbb{Z}/2)^N \rightarrow A
> $$
>
> from $(\mathbb{Z}/2)^N$ to a set $A$. In addition, we are promised
> that there is a secret bit-string $\mathbf{s}\in (\mathbb{Z}/2)^N$ such that for all
> $\mathbf{x},\mathbf{x}'\in(\mathbb{Z}/2)^N$,
> $f(\mathbf{x}) = f(\mathbf{x}')$ if and only if $\mathbf{x}' = \mathbf{x}$ or
> $\mathbf{x}' = \mathbf{x} \oplus \mathbf{s}$. The challenge is to find $\mathbf{s}$
> with as few calls as possible to the black box for $f$.

To get some intuition for the role of the secret bit-string $\mathbf{s}$,
consider the following function on $(\mathbb{Z}/2)^3$ where $A = (\mathbb{Z}/2)^2$:

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

It is easy to check that in this case the solution to Simon's problem is
$\mathbf{s}=101$. Note that $f$ is a 2 to 1 function. In general, we can think
of $\mathbf{s}$ as a bitmask where $f(\mathbf{x}) = f(\mathbf{x}')$ if and only
if $\mathbf{x}$ and $\mathbf{x}'$ differ exactly at the elements masked by
$\mathbf{s}$.

How hard is it to solve Simon's problem classically? Without knowing anything
about $f$, we intuitively have to keep calculating $f$ for different elements
$\mathbf{x}\in B_N$ until we get the same result twice. Based on the analysis of
the
[Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem#Arbitrary_number_of_days),
we have to compute $f$ for approximately $\sqrt{|(\mathbb{Z}/2)^N|}=2^{N/2}$ elements in
order to have a good chance of finding a collision.

Simon's algorithm, introduced in the
[same paper](https://epubs.siam.org/doi/10.1137/S0097539796298637), is a quantum
algorithm that solves Simon's problem in $\mathcal{O}(N)$ time - a nearly exponential
improvement over what is possible classically.

We'll now formulate Simon's problem as an instance of the
HSP. Consider the subgroup $H\subset (\mathbb{Z}/2)^N$ defined by:

$$
H = \{ \mathbf{0},\,\mathbf{s} \}
$$

where $\mathbf{s}$ is the hidden bit-string in Simon's problem. It is easy to see that 
$f$ [hides](#defn:hiding-a-subgroup) $H$. Therefore, solving the HSP for 
$G=(\mathbb{Z}/2)^N$ and the function $f$ is equivalent to solving Simon's problem.

As we will see in detail below, this means that (a generalization of)
Shor's algorithm  can be used to derive Simon's algorithm.

## Discrete Logarithms

Let $p$ be a prime number and let $\mathbb{Z}_p^\times$ denote the
[multiplicative group](https://en.wikipedia.org/wiki/Multiplicative_group_of_integers_modulo_n)
of integers module $p$.

Let $g\in\mathbb{Z}_p^\times$ be a [primitive root](https://en.wikipedia.org/wiki/Primitive_root_modulo_n)
of $\mathbb{Z}_p^\times$. This means $\mathbb{Z}_p^\times$ is generated by powers of $g$.

Now let $x\in\mathbb{Z}\_p^\times$ be an integer modulo $p$.
The _discrete logarithm_ of $x$ in base $g$ is defined to be the integer $0\leq s < p$
satisfying:

$$
x = g^s\ (\mathrm{mod}\ p)
$$

The discrete logarithm problem is, given $g$ and $x$, to find the discrete logarithm of $x$ in base $g$.

In general there is no known efficient solution to the discrete logarithm problem.
Indeed, this problem is the foundation of the
[Diffie-Hellman key exchange](https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange)
protocol in cryptography. An efficient method to compute discrete logarithms would
break this commonly used protocol.

Let's now see how to formulate the discrete logarithm problem as an instance of the HSG.

Consider the group $G=\mathbb{Z}/(p-1)\times\mathbb{Z}/(p-1)$ and the function:

$$
\begin{align*}
f: G &\rightarrow \mathbb{Z}_p^\times \\
(a, b) &\mapsto g^a x^{-b}
\end{align*}
$$

We claim that finding $s$, the discrete logarithm of $x$, can be reduced to solving the
HSP for the group $G$ with the function $f$.

First, note that $f$ is a group homomorphism. Furthermore, an element $(a,b)$ is in
the kernel of $f$, denoted $\mathrm{Ker}(f)$ if and only if:

$$
\begin{equation}\label{eq:dilog-kernel}
g^a = x^b\ (\mathrm{mod}\ p)
\end{equation}
$$

If $s$ is equal to the discrete logarithm of $x$ then by definition:

$$
\begin{equation}\label{eq:dilog-def}
x = g^s\ (\mathrm{mod}\ p)
\end{equation}
$$

Combining equations \ref{eq:dilog-kernel} and \ref{eq:dilog-def} implies that 
$(a,b)\in\mathrm{Ker}(f)$ if and only if:

$$
a = b \cdot s\ (\mathrm{mod}\ p - 1)
$$

In other words, the subgroup 

$$
H := \mathrm{Ker}(f)\subset\mathbb{Z}/(p-1)\times\mathbb{Z}/(p-1)
$$ 

is generated by the element $(s, 1)$:

$$
H = \langle (s, 1) \rangle
$$

Since $f$ is a homomorphism, it is easy to see that $f$ 
[hides](#defn:hiding-a-subgroup) its kernel $H$.

Therefore, a solution to the HSP would in particular allow us to find
a non trivial element $(bs, b) \in H$ from which we can easily compute $s$.

## The Standard Method

In this section we'll introduce a quantum algorithm for the HSP that is 
colloquially known as the _standard method_.

We'll start with some notation. For a finite group $G$, we will use 
$\mathbb{C}[G]$ to denote the vector space with one basis
vector for each element of $G$.
To be consistent with quantum mechanics notation,
we will denote the basis vector corresponding to $g$ by $|g\rangle$.
Concretely, elements of $\mathbb{C}[G]$ are of the form

$$
\sum_{g\in G}\alpha_g |g\rangle
$$

for complex coefficients $\alpha_g\in\mathbb{C}$.

Now, suppose that $f:G\rightarrow A$ is a function that hides a subgroup $H\subset G$.

The first step of the standard method is to prepare a superposition
of the basis vectors of $\mathbb{C}[G]$:

$$
\frac{1}{\sqrt{|G|}}\sum_{g\in G}|g\rangle
$$

We then apply the black box of $f$ to this state to obtain:

$$
\frac{1}{\sqrt{|G|}}\sum_{g\in G}|g\rangle|f(g)\rangle
$$

Next we measure the right register to obtain $f(g) \in A$ for a uniformly random
$g\in G$. After the measurement, the resulting state will be a sum over the 
vectors $|g'\rangle$ for which $f(g') = f(g)$. Since $f$ hides $H$, this is equal
to the sum over the coset $gH$:

$$
\frac{1}{\sqrt{|H|}}\sum_{h\in H}|gh\rangle
$$

To facilitate notation, we will use $|H\rangle$ to denote the normalized sum over the elements
of the subgroup $H$:

$$
|H\rangle := \frac{1}{\sqrt{|H|}}\sum_{h\in H}|h\rangle
$$

and we will similarly denote the sum over the coset $gH$ by $\|gH\rangle$:

$$
|gH\rangle := \frac{1}{\sqrt{|H|}}\sum_{h\in H}|gh\rangle
$$

With this notation, we can say that the first step of the standard method is 
to use a single invocation  of $f$ to produce a coset state

$$
|gH\rangle
$$

where $g\in G$ is chosen uniformly at random and $H$ is the subgroup hidden by $f$.

It is not immediately obvious how $|gH\rangle$ can be used to learn anything about $H$.
If we simply measure $|gH\rangle$ in the standard basis, the result will be $gh$ for some
$h\in H$. Since $g$ is uniformly distributed in $G$, so is $gh$. Therefore,
measuring $gh$ does not reveal any information about $H$.

The key to the second step of the standard method is to transform $|gH\rangle$
into another basis that in some sense removes the obfuscating effect of $g$ and
allows us to obtain information about $H$.

To see how this works, we'll first introduce the _shift operators_ on 
$\mathbb{C}[G]$:

{: #defn:shift-operator }

> **Definition (Shift Operator).** Let $G$ be a finite group and $g\in G$ an
> element of $G$. The _shift operator_ $L_g$ is the linear transformation
> of $\mathbb{C}[G]$ defined on a basis element $|g'\rangle$ by:
>
> $$
> L_g|g'\rangle := |gg'\rangle
> $$

The relevance to our situation is that we can rewrite the a coset state
$|gH\rangle$ in terms of the subgroup state $|H\rangle$ and the shift operator
$L_g$:

$$
|gH\rangle = L_g|H\rangle
$$

The key to the standard method is to apply a unitary transformation that
_simultaneously diagonalizes the shift operators $L_g$ for all $g\in G$_.
This is possible when $G$ is abelian since in that case the shift operators
commute.

The motivation for finding such a transformation is that, in general, an operator
that is diagonal in a given basis does not affect measurements in that basis.
Therefore, it's reasonable to assume that after moving to a basis that diagonalizes the
shift operators $L_g$, measuring $L_g|H\rangle$ will reveal direct information about
$H$.

In the next section we will introduce the _quantum Fourier transform_ (QFT) as an
explicit construction of a unitary transform that diagonalizes the shift operators.

In later sections we will apply it to the coset states obtained in our two examples
of the HSP, [Simons' Problem](#simons-problem) and the [Discrete Logarithm](#discrete-logarithms),
and recover the famously efficient quantum algorithms for those problems.


# The Quantum Fourier Transform

Recall from the [previous section](#the-standard-method) that our strategy for 
solving the hidden subgroup problem
is to find a unitary transformation that diagonalizes all of the shift operators
$L_g$ for $g\in G$. This transform is commonly called the
[quantum Fourier transform](https://en.wikipedia.org/wiki/Quantum_Fourier_transform) (QFT).

In this section we'll derive the QFT and then see precisely why it is so useful in the
context of the hidden subgroup problem.

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

What can we say about this function?
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
[characters](<https://en.wikipedia.org/wiki/Character_(mathematics)>).

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

We can restate the discussion above as saying that the eigenvalue function $\lambda$
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
> 3. $\chi(g^{-1}) = \chi(g)^*$ for all $g\in G$.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

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

3. It's easy to see that

   $$
   \chi(g^{-1}) = \chi(g)^{-1}
   $$

   By the previous item, $\chi(g)$ is a root of unity and so

   $$
   \chi(g)^{-1} = \chi(g)^*
   $$

_q.e.d_

</div>
</details>

Since this post is focused on abelian groups, it will be helpful to have an explicit
description of characters in that setting. 

Recall that by the
[Fundamental Theorem of Abelian Groups](https://en.wikipedia.org/wiki/Finitely_generated_abelian_group#Primary_decomposition),
any abelian group is a product of cyclic groups. We'll start by describing the characters on
cyclic groups and then show how characters on a product can be understood in terms of the
constituent factors.

{: #clm:cyclic-character-group }

> **Claim (Cyclic Character Group).** Let $N$ be an integer and let $\omega_N\in\mathbb{C}$ be the primitive
> $N$-th root of unity defined by 
>
> $$
> \omega_N = e^{2\pi i/N}
> $$
> 
> For each integer $0 \leq k < N$,
> let $\chi_k: \mathbb{Z}/N \rightarrow \mathbb{C}^\times$ be the function defined by:
>
> $$
> \chi_k(l) = \omega_N^{kl}
> $$
>
>Then:
>
> 1. For all $0 \leq k < N$ the function $\chi_k$ is a character on $\mathbb{Z}/N$
> 1. The following function is a group isomorphism between the cyclic group $\mathbb{Z}/N$
> and its group of characters:
>
> $$
> \begin{align*}
> F: \mathbb{Z}/N &\rightarrow \widehat{\mathbb{Z}/N} \\
> k &\mapsto \chi_k
> \end{align*}
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

The fact that $\chi_k$ is a character follows directly from the definition of a character.

To see that $F$ is a homomorphism, we must show that $\chi_{k_1 + k_2} = \chi_{k_1}\cdot\chi_{k_2}$
for all $k_1,k_2\in\mathbb{Z}/N$.

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

</div>
</details>

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

{: #clm:product-character-group}

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

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

It is easy to see that $P$ is a homomorphism. To see that $P$ is injective,
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

</div>
</details>

The next claim follows immediately from the previous two claims and the
[Fundamental Theorem of Abelian Groups](https://en.wikipedia.org/wiki/Finitely_generated_abelian_group#Primary_decomposition).

{: #clm:abelian-character-group}

> **Claim (Abelian Character Group).** For any finite abelian group $G$, $G\cong\hat{G}$.

## Eigenvectors

In the previous section we saw that the eigenvalues of $L_g$ are given by characters
on $G$. Interestingly, given a character $\chi\in\hat{G}$, it is easy to construct
a simultaneous eigenvector for $L_g$ whose eigenvalues are given by $\chi$:

{: #clm:eigenvectors-from-characters }

> **Claim (Eigenvectors From Characters).** Let $G$ be a finite group and let
> $\chi: G \rightarrow \mathbb{C}^\times$ be a character on $G$. Define
> $|\varphi\rangle\in\mathbb{C}[G]$ by:
>
> $$
> |\varphi\rangle = \frac{1}{\sqrt{|G|}}\sum_{g\in G}\chi(g)^*|g\rangle
> $$
>
> Then, for all $g\in G$, $|\varphi\rangle$ is an eigenvector of
> $L_g$ with eigenvalue $\chi(g)$.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

Recall that by [Character Properties](#clm:character-properties), 
$\chi^*(g)=\chi(g^{-1})$ for all $g\in G$. The claim follows by direct calculation:

$$
\begin{align*}
L_g|\varphi\rangle &= L_g\sum_{h\in G}\chi(h)^*|h\rangle \\
&= \frac{1}{\sqrt{|G|}}\sum_{h\in G}\chi(h)^*|gh\rangle \\
&= \frac{1}{\sqrt{|G|}}\sum_{h\in G}\chi(g^{-1}h)^*|g g^{-1}h\rangle \\
&= \frac{1}{\sqrt{|G|}}\sum_{h\in G}\chi(g^{-1})^*\chi(h)^*|h\rangle \\
&= \frac{1}{\sqrt{|G|}}\chi(g)\cdot\sum_{h\in G}\chi(h)^*|h\rangle \\
&= \chi(g)|\varphi\rangle
\end{align*}
$$

_q.e.d_

</div>
</details>

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
> **Claim (Diagonalization).** Let $G$ be a finite abelian group and 
> let $U$ be the linear transformation defined by:
>
> $$
> \begin{align*}
> U : \mathbb{C}[\hat{G}] &\rightarrow \mathbb{C}[G] \\
> |\chi\rangle &\mapsto \frac{1}{\sqrt{|G|}}\sum_{g\in G}\chi(g)^*|g\rangle
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

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

By [Eigenvectors From Characters](#clm:eigenvectors-from-characters), $U$ maps basis
vectors of $\mathbb{C}[\hat{G}]$ to simultaneous eigenvectors of $L_g$. Therefore,
it suffices to show that $U$ is unitary.

By [Abelian Character Group](#clm:abelian-character-group), $\mathbb{C}[\hat{G}]$
and $\mathbb{C}[G]$ have the same dimension. Therefore, to prove that $U$ is unitary it
is sufficient to show that it
maps basis vectors $|\chi\rangle\in\mathbb{C}[\hat{G}]$ to orthonormal vectors in $\mathbb{C}[G]$.

We'll start by showing orthogonality. Let $\chi_1\neq\chi_2\in\hat{G}$ be different characters on $G$.
Since $\chi_1\neq\chi_2$, there must be some $g\in G$ such that $\chi_1(g)\neq\chi_2(g)$.
By claim [Eigenvectors From Characters](#clm:eig-from-char), this implies that $U|\chi_1\rangle$
and $U|\chi_2\rangle$ are eigenvectors of $L_g$ with different eigenvalues. Since $L_g$ is a permutation
transformation, [it follows](https://en.wikipedia.org/wiki/Permutation_matrix#Linear-algebraic_properties)
that it is [normal](https://en.wikipedia.org/wiki/Normal_matrix). Eigenvectors of a normal transformation
with distinct eigenvalues are orthogonal which implies that $U|\chi_1\rangle$ and $U|\chi_2\rangle$
are orthogonal.

It remains to show that $U|\chi\rangle$ has unit length for all $\chi\in\hat{G}$.
Recall that by [Character Properties](#clm:character-properties), $\chi(g)$ is a root of unity for all
$g\in G$ which implies that $\|\chi^*(g)\|^2 = 1$. Therefore:

$$
||U|\chi\rangle||^2 = \frac{1}{G}\sum_{g\in G}|\chi^*(g)|^2 = 1
$$

_q.e.d_

</div>
</details>

We will define the _quantum Fourier transform_ of $G$ to be the complex conjugate of $U$:

{: #defn:quantum-fourier-transform }

> **Definition (Quantum Fourier Transform).** Let $G$ be a finite abelian group.
> The _quantum Fourier transform_ (QFT) of $G$ is defined by:
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

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

This follows immediately from [Diagonalization](#clm:unitary-diag) and the fact that
$\mathrm{QFT} = U^*$

_q.e.d_

</div>
</details>

It is common to construct groups as products of smaller groups. Suppose that $G$ and $H$
are finite abelian groups and that we know $\mathrm{QFT}\_G$ and $\mathrm{QFT}\_H$. What can we say about
$\mathrm{QFT}_{G\times H}$, the QFT of the product $G\times H$?

First of all, note that $\mathbb{C}[G\times H]$ is isomorphic to $\mathbb{C}[G]\otimes\mathbb{C}[H]$
with the isomorphism given by:

$$
\begin{align*}
\mathbb{C}[G]\otimes\mathbb{C}[H] &\rightarrow \mathbb{C}[G\times H] \\
|g\rangle \otimes |h\rangle &\mapsto |g,h\rangle
\end{align*}
$$

We will therefore slightly abuse notation and identify $\mathbb{C}[G\times H]$ with
$\mathbb{C}[G]\otimes\mathbb{C}[H]$. We can use this identification to compare
$\mathrm{QFT}\_{G\times H}$ to $\mathrm{QFT}\_{G}$ and $\mathrm{QFT}\_{H}$

{: #clm:qft-product }

> **Claim (QFT Product).** Let $G$ and $H$ be finite abelian groups. Then:
>
> $$
> \mathrm{QFT}_{G\times H} = \mathrm{QFT}_G \otimes \mathrm{QFT}_H
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

First, let's see how the characters on $G\times H$ relate to the characters on
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
\mathrm{QFT}_{G\times H}|g,h\rangle &= \frac{1}{\sqrt{|G\times H|}}\sum_{\mu\in \widehat{G\times H}}\mu(g,h)|\mu\rangle \\
&= \frac{1}{\sqrt{|G|\cdot| H|}}\sum_{\chi\in \hat{G},\,\lambda\in\hat{H}}\chi(g)\lambda(h)|\chi\rangle|\lambda\rangle \\
&= \left(\frac{1}{\sqrt{|G|}}\sum_{\chi\in \hat{G}}\chi(g)|\chi\rangle\right) \otimes \left(\frac{1}{\sqrt{|H|}}\sum_{\lambda\in \hat{H}}\lambda(h)|\lambda\rangle\right) \\
&= \mathrm{QFT}_G|g\rangle \otimes \mathrm{QFT}_H|h\rangle
\end{align*}
$$

_q.e.d_

</div>
</details>

## Coset States

We'll now apply the quantum Fourier transform to the hidden subgroup problem.
Recall from section [The Standard Method](#the-standard-method)
that the first part of the standard method is to produce a coset state:

$$
|gH\rangle = \frac{1}{\sqrt{|H|}}\sum_{h\in H}|gh\rangle
$$

where $H$ is the hidden subgroup and $g$ is a uniformly random element of $G$. We'll use 
$\mathrm{QFT}_G$
to extract information about $H$ from the coset state $|gH\rangle$.

If $\chi\in\hat{G}$ is a character on $G$ and $H\subset G$ is a subgroup of $G$
then we will use the notation $\chi_H$ to denote the restriction of $\chi$ to $H$.

As a warmup, let's see what happens if we apply $\mathrm{QFT}_G$ to the subgroup state
$\|H\rangle$:

{: #clm:qft-subgroup-state }

> **Claim (QFT Subgroup State).** Let $G$ be a finite abelian group and let $H\subset G$ be
> a subgroup. Then:
>
> $$
> \mathrm{QFT}_G(|H\rangle) = \sqrt{\frac{|H|}{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}|\chi\rangle
> $$

In other words, applying $\mathrm{QFT}_G$ to $|H\rangle$ gives a sum over the characters on $G$
whose restriction to $H$ is trivial.

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

First, note that for all $h\in H$ the state $\|H\rangle$ is invariant under
a shift by $h$:

$$
\begin{equation}\label{eq:subgroup-state-shift}
L_h|H\rangle = |H\rangle
\end{equation}
$$

By [QFT Diagonalization](#clm:qft-diagonalization), $\mathrm{QFT}_G$ is unitary and:

$$
L_h = \mathrm{QFT}_G^* \circ D_h \circ \mathrm{QFT}_G
$$

Applying this to \ref{eq:subgroup-state-shift} and multiplying both sides
by $\mathrm{QFT}_G$ gives:

$$
\mathrm{QFT}_G|H\rangle = D_h\mathrm{QFT}_G|H\rangle 
$$

Taking the inner product of both sides with a character $\chi\in\hat{G}$ and
using the definition of $D_h$:

$$
\begin{align*}
\langle\chi|\mathrm{QFT}_G|H\rangle &= \langle\chi|D_h\mathrm{QFT}_G|H\rangle \\
&= \chi(h)\langle\chi|\mathrm{QFT}_G|H\rangle
\end{align*}
$$

Therefore, either $\langle\chi\|\mathrm{QFT}_G\|H\rangle=0$ or $\chi_H=1$.

Suppose that $\chi_H=1$. Then:

$$
\begin{align*}
\langle\chi|\mathrm{QFT}_G|H\rangle &= \frac{1}{\sqrt{|G|}\sqrt{|H|}}\sum_{h\in H}\chi(h) \\
&= \frac{1}{\sqrt{|G|}\sqrt{|H|}}\sum_{h\in H}1 \\
&= \sqrt{\frac{|H|}{|G|}}
\end{align*}
$$

_q.e.d_

</div>
</details>

We can use this claim together with another application of [QFT Diagonalization](#clm:qft-diagonalization) to determine what
happens when we apply $\mathrm{QFT}_G$ to a coset state:

{: #clm:qft-coset-state }

> **Claim (QFT Coset State).** Let $G$ be a finite abelian group, $H\subset G$ be
> a subgroup and $g\in G$ an element of $G$. Then:
>
> $$
> \mathrm{QFT}_G(|gH\rangle) = \sqrt{\frac{|H|}{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}\chi(g)|\chi\rangle
> $$

<details>
<summary>
Proof [click to expand]
</summary>
<div class="details-content">

We can rewrite the coset state as:

$$
|gH\rangle = L_g|H\rangle
$$

By [QFT Diagonalization](#clm:qft-diagonalization), $L_g = \mathrm{QFT}_G^*\circ D_g\circ \mathrm{QFT}_G$
and so:

$$
\mathrm{QFT}_G|gH\rangle = (D_g\circ\mathrm{QFT}_G)|H\rangle
$$

Applying [QFT Subgroup State](#clm:qft-subgroup-state) gives:

$$
\begin{align*}
\mathrm{QFT}_G|gH\rangle &= D_g \frac{\sqrt{|H|}}{\sqrt{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}|\chi\rangle \\
&= \frac{\sqrt{|H|}}{\sqrt{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}D_g|\chi\rangle \\
&= \frac{\sqrt{|H|}}{\sqrt{|G|}}\sum_{\chi\in\hat{G},\,\chi_H=1}\chi(g)|\chi\rangle 
\end{align*}
$$

_q.e.d_


</div>
</details>

This gives us a strategy for extracting information about $H$ from an arbitrary coset state $|gH\rangle$.
According to [QFT Coset State](#clm:qft-coset-state), if we apply $\mathrm{QFT}_G$
to $|gH\rangle$ and measure in
the standard basis of $\mathbb{C}[\hat{G}]$ then we are guaranteed to measure a character $\chi$
whose restriction to $H$ is trivial. As we will see in the following sections, this essentially imposes
a linear constraint on the generators of $H$. Repeating the process $\mathcal{O}(\log\|H\|)$ times will determine
the generators of $H$ with high probability.

## QFT Implementation

In order for the QFT to be useful, we need to be able to implement it efficiently
with a quantum circuit. By _efficiently_ we mean that the quantum circuit for the QFT
of the group $G$ should consist of $\mathcal{O}(\log |G|)$ 
[gates](https://en.wikipedia.org/wiki/Quantum_logic_gate).

As a first step, by the
[Fundamental Theorem of Abelian Groups](https://en.wikipedia.org/wiki/Finitely_generated_abelian_group#Primary_decomposition),
and claim [QFT Product](#clm:qft-product), it is sufficient to efficiently implement
the QFT on the cyclic groups $\mathbb{Z}/N$ for an arbitrary integer $N$.

In the next [section](#qft-implementation-1) we'll see how to do this in the case where $N=2$.
Deriving an efficient circuit for an arbitrary $N$ is outside the scope of this
post, but a good reference is chapter 4 of 
[Andrew Childs' Lecture Notes](https://www.cs.umd.edu/~amchilds/qa).


# Simon's Algorithm

In section [Simon's Problem](#simons-problem) we introduced Simon's problem and 
saw that it is a special case of
the hidden subgroup problem with

$$
G = (\mathbb{Z}/2)^N
$$

and function $f:G\rightarrow A$ that hides a subgroup $H\subset G$ of order $2$.

In this section we'll apply the theory from section [The Quantum Fourier Transform](#the-quantum-fourier-transform)
to this group and recover Simon's original solution to the problem.

We'll represent elements of $(\mathbb{Z}/2)^N$ as bit-strings 

$$
\mathbf{x} = (x_1,\dots,x_n)
$$

where $x_i\in\{0,1\}$. The group operation on $(\mathbb{Z}/2)^N$ is bitwise addition modulo $2$
which we will denote by $\oplus$.

## Characters
 
In order to construct the quantum Fourier transform we need to
find the characters on $(\mathbb{Z}/2)^N$.

We'll start by finding the characters on $\mathbb{Z}/2$. 
By claim [Cyclic Character Group](#clm:cyclic-character-group), the only
characters on $\mathbb{Z}/2$ are $\chi_0$ and $\chi_1$ defined by:

$$
\begin{align*}
\chi_0(x) &= 1 \\
\chi_1(x) &= (-1)^x
\end{align*}
$$

where $x\in\mathbb{Z}/2$.

Therefore, by [Product Character Group](#clm:product-character-group)
it follows that all of the characters on $(\mathbb{Z}/2)^N$ are of the form:

$$
\begin{equation}\label{eq:bitstring-char}
\chi(\mathbf{x}) = \prod_{i=1}^N\chi_{y_i}(x_i)
\end{equation}
$$

for some $y_1,\dots,y_N\in\{0,1\}$.

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

Note that if $\mathbf{y}=(y_1,\dots,y_N)$ then we can rewrite equation \ref{eq:bitstring-char}
as:

$$
\prod_{i=1}^N\chi_{y_i}(x_i) = \chi_\mathbf{y}(\mathbf{x})
$$

It therefore follows that the characters on $(\mathbb{Z}/2)^N$
are all of the form $\chi_\mathbf{y}$ for some $\mathbf{y}\in(\mathbf{Z}/2)^N$.

By [definition](#defn:quantum-fourier-transform), this implies that the QFT on this group is given by:

$$
\mathrm{QFT}_{(\mathbf{Z}/2)^N}(|\mathbf{x}\rangle) = 
\frac{1}{2^{N/2}}\sum_{\mathbf{y}\in(\mathbb{Z}/2)^N}(-1)^{\mathbf{x}\cdot\mathbf{y}}|\mathbf{y}\rangle
$$

Note that in this sum, we are representing the character basis elements $\|\chi_\mathbf{y}\rangle$ by
the bit-string $\|\mathbf{y}\rangle$.

## The Standard Method

We now use the [standard method](#the-standard-method) together with
$\mathrm{QFT}_{(\mathbf{Z}/2)^N}$ so solve Simon's problem.

First, [recall](#simons-problem) that in this case the hidden subgroup $H$ is 
guaranteed to be of order $2$. So we can write it as:

$$
H = \{\mathbf{0}, \mathbf{s} \}
$$

for some hidden element $\mathbf{s}\in(\mathbb{Z}/2)^N$ that we are trying to find.

As usual, the first step of the standard method produces a coset state:

$$
|\mathbf{x} \oplus H\rangle = \frac{1}{\sqrt{2}}(|\mathbf{x}\rangle + |\mathbf{x}\oplus\mathbf{s}\rangle)
$$

for some uniformly random $\mathbf{x}\in(\mathbb{Z}/2)^N$.

By [QFT Coset State](#clm:qft-coset-state), if we apply $\mathrm{QFT}_{(\mathbf{Z}/2)^N}$
to this coset state we will get:

$$
\mathrm{QFT}_{(\mathbb{Z}/2)^N}(|\mathbf{x} \oplus H\rangle) =
\frac{1}{2^{(N-1)/2}}\sum_{\mathbf{y}\in(\mathbb{Z}/2)^N,\ \chi_\mathbf{y}(\mathbf{s})=1}\chi_\mathbf{y}(\mathbf{x})|\mathbf{y}\rangle
$$


This means that if we measure $\mathrm{QFT}_{(\mathbb{Z}/2)^N}(|\mathbf{x} \oplus H\rangle)$ in the standard basis
we'll get an element $\mathbf{y}\in(\mathbb{Z}/2)^N$ satisfying

$$
\chi_\mathbf{y}(\mathbf{s}) = 1
$$

By the definition of $\chi_\mathbf{y}$ this is equivalent to:

$$
\mathbf{y}\cdot\mathbf{s} = 0
$$

In other words, after measuring $\mathrm{QFT}_{(\mathbb{Z}/2)^N}(|\mathbf{x} \oplus H\rangle)$
we will get one linear constraint on the hidden vector $\mathbf{s}$.
If we repeat the process $\mathcal{O}(N)$ times, we have a high
chance of finding $N$ independent linear constraints on the length $N$ vector $\mathbf{s}$ which
is sufficient to recover $\mathbf{s}$ and solve Simon's problem.

## QFT Implementation

In this section we'll show how to efficiently implement the QFT on $(\mathbb{Z}/2)^N$.
By _efficiently_ we mean that the quantum circuit for the QFT should consist of $\mathcal{O}(N)$ 
[quantum gates](https://en.wikipedia.org/wiki/Quantum_logic_gate).

Specifically, we will implement $\mathrm{QFT}_{(\mathbb{Z}/2)^N}$ using $N$ 
[Hadamard gates](https://en.wikipedia.org/wiki/Quantum_logic_gate#Hadamard_gate).

Recall that a Hadamard gate $H$ on a qubit is defined by:

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

where $x\in\mathbb{Z}/2$. But note that this is equal to the QFT on $\mathbb{Z}/2$:

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

In section [Discrete Logarithms](#discrete-logarithms) we saw that 
computing discrete logarithms can be reduced to solving the hidden subgroup problem on the group 

$$
G=\mathbb{Z}/N\times\mathbb{Z}/N
$$

for an integer $N$.

In this section we'll
compute the QFT for $\mathbb{Z}/N \times \mathbb{Z}/N$ and use it to derive Shor's algorithm for the
discrete logarithm. 

## Characters

In order to compute $\mathrm{QFT}_{\mathbb{Z}/N \times \mathbb{Z}/N}$ we need to
describe the characters on $\mathbb{Z}/N \times \mathbb{Z}/N$.

Let $\omega_N\in\mathbb{C}$ be the primitive $N$-th root of unity defined by:

$$
\omega_N = e^{2\pi i/N}
$$

For a pair $(k_1,k_2)\in\mathbb{Z}/N\times\mathbb{Z}/N$,
let $\chi_{k_1,k_2}: \mathbb{Z}/N\times\mathbb{Z}/N \rightarrow\mathbb{C}^\times$ be the
function defined by:

$$
\chi_{k_1,k_2}(l_1,l_2) = \omega^{k_1l_1 + k_2l_2}
$$

By claims [Cyclic Character Group](#clm:cyclic-character-group) and
[Product Character Group](#clm:product-character-group), all of the characters on
$\mathbb{Z}/N\times\mathbb{Z}/N$ are equal to $\chi_{k_1,k_2}$ for some
$(k_1,k_2)\in\mathbb{Z}/N\times\mathbb{Z}/N$.

By [definition](#defn:quantum-fourier-transform), it follows that the QFT on
$\mathbb{Z}/N\times\mathbb{Z}/N$ is given by:

$$
\mathrm{QFT}_{\mathbb{Z}/N\times\mathbb{Z}/N}(|l_1,l_2\rangle) 
= \frac{1}{N}\sum_{k_1=0}^{N-1}\sum_{k_2=0}^{N-1}\omega_N^{k_1l_1 + k_2l_2}|k_1,k_2\rangle
$$

## The Standard Method

In section [Discrete Logarithms](#discrete-logarithms) we saw that finding the 
discrete logarithm of $x$ can be reduced to solving the hidden subgroup
problem on $\mathbb{Z}/(p-1)\times\mathbb{Z}/(p-1)$ for a subgroup of the form:

$$
H = \langle (s, 1) \rangle \subset \mathbb{Z}/(p-1)\times\mathbb{Z}/(p-1)
$$

for some secret $s\in\mathbb{Z}/(p-1)$.

By following the standard method, we can produce a coset state

$$
|(a, 0) + H\rangle = \sum_{b\in\mathbb{Z}/(p-1)}|(a+bs, b)\rangle
$$

For some $a\in\mathbb{Z}/(p-1)$.
The problem is now reduced to determining $s\in\mathbb{Z}/(p-1)$ based on states of this form.

By [QFT Coset State](#clm:qft-coset-state) and our description of the characters on
$\mathbb{Z}/(p-1)\times\mathbb{Z}/(p-1)$ in the previous section, if we apply 
$\mathrm{QFT}\_{\mathbb{Z}/(p-1)\times\mathbb{Z}/(p-1)}$ to the coset state $|(a, 0) + H\rangle$
and measure in the standard basis we will get some pair

$$
(k_1,k_2)\in\mathbb{Z}/(p-1)\times\mathbb{Z}/(p-1)
$$

such that the restriction of $\chi_{k_1,k_2}$ to $H$ is trivial. Since $H$
is generated by $(s,1)$, this condition is equivalent to:

$$
\chi_{k_1,k_2}(s, 1) = \omega_{p-1}^{k_1\cdot s + k_2} = 1
$$

which in turn implies that $(k_1,k_2)$ must satisfy:

$$
k_1\cdot s + k_2 = 0\ (\mathrm{mod}\ p-1)
$$

If $k_1$ is coprime to $p-1$, then it has a multiplicative inverse modulo $p-1$ and
we can solve for $s$.

If $k_1$ is not coprime to $p-1$ then we just repeat the process again. By the
[growth rate of Euler's totient function](https://en.wikipedia.org/wiki/Euler%27s_totient_function#Growth_rate),
we will not have to repeat this process many times before finding a pair $(k_1,k_2)$ where
$k_1$ is coprime to $p-1$. 


























