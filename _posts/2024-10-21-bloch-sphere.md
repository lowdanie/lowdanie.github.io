---
layout: post
title: "Understanding The Bloch Sphere"
date: 2024-10-21
mathjax: true
utterance-issue: 9
---

# Introduction

A single [qubit](https://en.wikipedia.org/wiki/Qubit) is a vector of the form

$$
| \psi \rangle = a |0\rangle + b |1\rangle
$$

where $a,b\in\mathbb{C}$ are complex numbers satisfying

$$
\langle \psi | \psi \rangle = |a|^2 + |b|^2 = 1
$$

In other words, the state space of a single qubit is the set of unit vectors in
the two dimensional complex vector space $\mathbb{C}^2$ with basis vectors
$|0\rangle$ and $|1\rangle$.

Operations on a single qubit are linear transformations which preserve the norm
and therefore correspond to $2\times 2$
[unitary matrices](https://en.wikipedia.org/wiki/Unitary_matrix) $U \in U(2)$.

The vector space $\mathbb{C}^2$ is a four dimensional real vector space, and the
constraint $|a|^2 + |b|^2 = 1$ means that the state space of a qubit can be
identified with the
[three dimensional sphere](https://en.wikipedia.org/wiki/3-sphere)
$S^3 \subset \mathbb{R}^4$. Concretely, if $a = x + yi$ and $b=z + wi$ then
$|a|^2=x^2+y^2$ and $|b|^2 = z^2 + w^2$ and so the state space of a qubit
consists of the points $(x,y,z,w)\in\mathbb{R}^4$ satisfying:

$$
x^2 + y^2 + z^2 + w^2 = 1
$$

which is precisely the definition of $S^3$.

It is difficult to visualize $S^3$ and even more difficult to imagine how it
behaves under unitary transformations.

The _Bloch Sphere_ is a projection of the state space onto the more familiar
[two dimensional sphere](https://en.wikipedia.org/wiki/Sphere)
$S^2 \subset \mathbb{R}^3$. Importantly, under this projection unitary
transformations of the state space $S^3$ correspond to ordinary rotations of the
sphere. Furthermore, there is an explicit formula that determines precisely
which rotation corresponds to a given unitary matrix. This makes the Bloch
sphere an indispensible tool for analyzing single qubit operations.

As we will see in the next section, the formula relating a unitary matrix to its
associated rotation is not very obvious. Furthermore, the standard proof of the
formula involves lengthy trigonometric calculations which, at least to me, are
not very illuminating.

The goal of this post is to present an alternative construction of the Bloch
sphere under which the rotation formula is quite intuitive and indeed can be
proved directly with essentially no calculation at all.

In the next section we'll review the standard definition of the Bloch sphere
together with the rotation formula alluded to above as can be found in many
places such as
[Nielsen and Chuang](https://www.google.com/books/edition/Quantum_Computation_and_Quantum_Informat/aai-P4V9GJ8C?hl=en&gbpv=0)
, [wikipedia](https://en.wikipedia.org/wiki/Qubit#Bloch_sphere_representation)
and this [online book](https://qubit.guide/index).

Next we'll present an alternative perspective and use it to provide a succinct
and intuitive proof of the formula.

# The Bloch Sphere

As stated in the introduction, a single qubit is a vector of the form

$$
| \psi \rangle = a |0\rangle + b |1\rangle
$$

where $a,b\in\mathbb{C}$ are complex numbers satisfying

$$
|a|^2 + |b|^2 = 1
$$

Due to this constraint, we can parameterize the qubit with three angles as:

$$
| \psi \rangle = e^{i \varphi_0}\cos(\theta/2) |0\rangle + e^{i\varphi_1}\sin(\theta/2) |1\rangle
$$

We can factor out a global phase $e^{i \varphi_0}$ and, after defining
$\varphi = \varphi_1 - \varphi_0$, rewrite this as:

$$
| \psi \rangle = e^{i \varphi_0} \left(\cos(\theta/2) |0\rangle + e^{i\varphi}\sin(\theta/2) |1\rangle \right)
$$

Recall that in
[spherical coordinates](https://en.wikipedia.org/wiki/Spherical_coordinate_system)
the pair of angles $(\theta, \varphi)$ define a point on the unit sphere:

{% include image_with_source.html url="/assets/bloch_sphere/Bloch_sphere.svg.png" source_url="https://commons.wikimedia.org/w/index.php?curid=5829358" %}

The Bloch sphere projection maps the qubit state $|\psi\rangle$ to the point on
the sphere with polar coordinates $(\theta,\varphi)$. In cartesian coordinates
this point is equal to:

$$
(\cos\varphi\sin\theta, \sin\varphi\sin\theta, \cos\theta) \in \mathbb{R}^3
$$

To facilitate notation, we will denote the state space of a qubit by
$S\subset\mathbb{C}^2$ and denote the Bloch projection by $\mathrm{Bloch}$:

$$
\mathrm{Bloch} : S \rightarrow \mathbb{R}^3
$$

Note that this projection ignores the global phase $e^{i \varphi_0}$. The
physical motivation for this is that due to the properties of
[quantum measurement](https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics)
a global phase has no observable effect.

## Unitary Transformations

Recall that a $2\times 2$
[unitary matrix](https://en.wikipedia.org/wiki/Unitary_matrix) $U$ is a
$2\times 2$ matrix with complex values satisfying

$$
U U^* = I
$$

where $U^*$ denotes the complex conjugate of $U$. The set of unitary matrices
together with matrix multiplication form a the
[Unitary Group](https://en.wikipedia.org/wiki/Unitary_group) denoted $U(2)$.

We can represent a qubit state $|\psi\rangle = a|0\rangle + b|1\rangle$ as a
length $2$ column vector in $\mathbb{C}^2$:

$$
|\psi\rangle = \left[\begin{matrix}a\\b\end{matrix}\right]
$$

We can then transform a qubit state $|\psi\rangle$ by a unitary matrix $U$ via
matrix multiplication:

$$
U|\psi\rangle = U\left[\begin{matrix}a\\b\end{matrix}\right]
$$

It is natural to wonder what multiplication by $U$ corresponds to on the Bloch
sphere. Or, more precisely, what is the relationship between the points
$\mathrm{Bloch}(|\psi\rangle)\in\mathbb{R}^3$ and
$\mathrm{Bloch}(U|\psi\rangle)\in\mathbb{R}^3$?

It turns out that transforming qubit states by a unitary matrix corresponds to
_rotating_ the Bloch sphere. Rotations in $\mathbb{R}^3$ can be parameterized by
an
[axis and an angle](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation).
There is a formula that we can use to find the axis and angle of the rotation of
the Bloch sphere that corresponds to a given unitary matrix.

The formula relies on the parameterization of unitary matrices by
[Pauli Matrices](https://en.wikipedia.org/wiki/Pauli_matrices) via the
[matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential) which we
will introduce in the following sections.

## The Special Unitary Group

It is easy to prove that the determinant of a unitary matrix $U\inU(2)$ is a
complex number with norm $1$. First, applying the determinant to both sides of
the equation $UU^* = I$ gives us:

$$
\mathrm{det}(UU^*) = det(I) = 1.
$$

By elementary properties of the determinant:

$$
\mathrm{det}(UU^*) = \mathrm{det}(U)\mathrm{det}(U)^\* = |\mathrm{det}(U)|^2
$$

Putting this together we see that indeed:

$$
|\mathrm{det}(U)| = 1
$$

The
[Special Unitary Group](https://en.m.wikipedia.org/wiki/Special_unitary_group),
denoted $\mathrm{SO}(2)$ is the subgroup of $U(2)$ consisting of matrices whose
determinant is _exactly_ $1$. The relationship we alluded to earlier between
unitary transformations and rotations of the Bloch sphere is only true for
unitary transformations whose determinant is equal to $1$ and so for the
remainder of this post we will focus on the special unitary group $SU(2)$.

As an aside, note that is similar to the situation with
[orthogonal matrices](https://en.wikipedia.org/wiki/Orthogonal_matrix) over the
real numbers which can have a determinant of $1$ or $-1$. Only the orthogonal
matrices with determinant $1$ preserve orientations and rotations of
$\mathbb{R}^3$ are therefore identified with the
[Special Orthogonal Group](https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group).

A useful tool for studying matrix groups such as $SU(2)$ is the
[matrix exponential](https://en.wikipedia.org/wiki/Matrix_exponential) which
sends an $n\times n$ matrix with complex coefficients $M$ to another $n\times n$
matrix $e^M$.

In the case where $n=1$, this is simply the usual exponential of a complex
number. This is generalized to any $n$ by plugging the matrix $M$ into the same
Taylor series that is used to define the scalar exponential:

$$
e^M = \sum_{i=0}^\infty \frac{1}{i!}M^i
$$

Suppose that $U = e^M$ is a special unitary matrix. What does this imply about
the matrix $M$? Using the definition of a unitary matrix we get:

$$
e^M \cdot \left(e^M\right)^* = UU^* = I
$$

By using basic properties of the exponential map we get:

$$
e^M \cdot \left(e^M\right)^* = e^M e^{M^\*} = e^{M + M^\*}
$$

Putting this together we get:

$$
e^{M + M^*} = I
$$

which implies that $M = -M^*$. I.e, $M$ must be _skew Hermitian_. Note that if
$M$ is Hermitian then $iM$ is skew Hermitian. Putting this all together we see
that if $M$ is a Hermitian matrix then $e^{iM}$ is unitary.

In addition to being unitary, by the definition of $\mathrm{SU}(2)$ we know that
$\mathrm{det}(U)=1$. The
[Jacobi Formula](https://en.wikipedia.org/wiki/Matrix_exponential#The_determinant_of_the_matrix_exponential)
relates the trace of a matrix $X$ to the determinant of $e^X$:

$$
\mathrm{det}(e^X) = e^{\mathrm{tr}(X)}
$$

Applying this to $U=e^{iM}$ gives us:

$$
e^{\mathrm{tr}(iM)} = \mathrm{det}(e^{iM}) = \mathrm{det}(U) = 1
$$

which implies That

$$
\mathrm{tr}(iM) = 0
$$

Since $\mathrm{tr}(iM) = i\mathrm{tr}(M)$ this implies that $\mathrm{tr}(M)=0$.

In summary, if $e^{iM}$ is a special unitary matrix then $M$ is a traceless
Hermitian matrix.

The same calculation shows that the converse is true as well. I.e, if $M$ is a
Hermitian matrix with trace $0$ then $e^{iM}$ is a special unitary matrix.

We will denote the set of $2\times 2$ Hermitian matrices with trace $0$ by
$\mathfrak{su}(2)$. Another way of stating the above correspondence between
$\mathfrak{su}(2)$ and $\mathrm{SU}(2)$ is that $\mathfrak{su}(2)$ is the
[lie algebra](https://en.wikipedia.org/wiki/Special_unitary_group#Lie_algebra_2)
of $\mathrm{SU}(2)$. Note that we are using the physics convention of working
with Hermitian matrices multiplied by $i$ rather than skew Hermitian matrices.

Since $\mathrm{SU}(2)$ is
[compact and connected](https://en.wikipedia.org/wiki/Special_unitary_group#Properties),
_any_ special unitary matrix $U$ has the form $e^{iM}$ for some
$M\in\mathfrak{su}(2)$. In other words, we can parameterize $\mathrm{SU}(2)$ by
matrices in $\mathfrak{su}(2)$.

It is easy to see that $\mathfrak{su}(2)$ is closed under matrix addition and
scalar multiplication which gives is the structure of a vector space. In the
next section we'll introduce the
[Pauli Matrices](https://en.wikipedia.org/wiki/Pauli_matrices) which form a
basis for $\mathfrak{su}(2)$.

## Pauli Matrices

In the previous section we introduced the vector space $\mathfrak{su}(2)$ of
$2\times 2$ traceless Hermitian matrices with complex coefficients. To be
concrete, consider the $2\times 2$ matrix

$$
M = \left[ \begin{matrix}a & b \\ c & d \end{matrix}\right]
$$

where $a,b,c,d \in \mathbb{C}$. Let's see what $M$ being in $\mathfrak{su}(2)$
implies for the coefficients of $M$.

The Hermitian condition means that $M=M^*$ and so

$$
\left[ \begin{matrix}a & b \\ c & d \end{matrix}\right] =
\left[ \begin{matrix}a^* & c^* \\ b^* & d^* \end{matrix}\right]
$$

This implies that $a$ and $d$ are real numbers and that $b = c^*$.

The traceless condition implies that:

$$
\mathrm{tr}(M) = a + d = 0
$$

and so $a = -d$. Setting $a=x\in\mathbb{R}$ and $c=z+yi\in\mathbb{C}$ we can
write $M$ as:

$$
M = \left[\begin{matrix}x & z-yi \\ z+yi & -x \end{matrix}\right] =
x \left[\begin{matrix}1 & 0 \\ 0 & -1 \end{matrix}\right] +
y \left[\begin{matrix}0 & -i \\ i & 0 \end{matrix}\right] +
z \left[\begin{matrix}0 & 1 \\ 1 & 0 \end{matrix}\right]
$$

for some _real_ numbers $x,y,z\in\mathbb{R}$. The matrices

$$
\begin{align*}
X &= \left[ \begin{matrix}1 & 0 \\ 0 & -1 \end{matrix}\right] \\
Y &= \left[ \begin{matrix}0 & -i \\ i & 0 \end{matrix}\right] \\
Z &= \left[ \begin{matrix}0 & 1 \\ 1 & 0 \end{matrix}\right]
\end{align*}
$$

are called [Pauli Matrices](https://en.wikipedia.org/wiki/Pauli_matrices). The
above calculation shows that the Pauli matrices form a basis for
$\mathfrak{su}(2)$.

The [Pauli Vector](https://en.wikipedia.org/wiki/Pauli_matrices#Pauli_vectors)
is defined to be the tuple:

$$
\overrightarrow{\sigma} = \left(X,Y,Z\right)
$$

Analogously to the dot product, the product of a vector $(x,y,z)\in\mathbb{R}^3$
and $\overrightarrow{\sigma}$

## The Rotation Formula

We are now ready to state the correspondence between special unitary matrices
$U\in\mathrm{SU}(2)$ and rotations of the Bloch sphere.

Let $U\in\mathrm{SU}(2)$ be a special unitary matrix. By our results in section
[The Special Unitary Group](#the-special-unitary-group), there exists a
traceless Hermitian matrix $H\in\mathfrak{su}(2)$ such that

$$
U = e^{iH}
$$

As we say in the previous section, there exist real numbers
$x,y,z \in \mathbb{R}$ such that

$$
H = xX +
$$
