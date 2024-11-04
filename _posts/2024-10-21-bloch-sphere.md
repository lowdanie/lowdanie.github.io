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

For example, consider the state:

$$
|\psi\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + |1\rangle\right)
$$

We can write this as

$$
|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\cdot \varphi}\sin(\theta/2)|1\rangle
$$

for $\theta = \pi/2$ and $\varphi = 0$. Therefore, by the definition of the
Bloch projection:

$$
\mathrm{Bloch}(|\psi\rangle) = (\cos(0)\sin(\pi/2), \sin(0)\sin(\pi/2), \cos(\pi/2)) =
(1, 0, 0)
$$

## Unitary Transformations

Recall that a $2\times 2$
[unitary matrix](https://en.wikipedia.org/wiki/Unitary_matrix) $U$ is a
$2\times 2$ matrix with complex values satisfying

$$
U U^* = I
$$

where $U^*$ denotes the complex conjugate of $U$. The set of unitary matrices
together with matrix multiplication form a the
[Unitary Group](https://en.wikipedia.org/wiki/Unitary_group) denoted
$\mathrm{U}(2)$.

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

For example, consider the state

$$
|\psi\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle + |1\rangle\right)
$$

and the unitary matrix:

$$
Z = \left[\begin{matrix} 1 & 0 \\ 0 & -1 \end{matrix}\right]
$$

Applying $Z$ to $\|\psi\rangle$ gives us:

$$
Z|\psi\rangle = \frac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle\right)
$$

All operations that we can physically apply to a qubit correspond to
transformation by a unitary matrix $U$. It is therefore natural to wonder what
multiplication by $U$ corresponds to on the Bloch sphere. Or, more precisely,
what is the relationship between the points
$\mathrm{Bloch}(|\psi\rangle)\in\mathbb{R}^3$ and
$\mathrm{Bloch}(U|\psi\rangle)\in\mathbb{R}^3$?

We'll start by answering this question for the unitary matrix $Z$ from the
previous example. If we apply $Z$ to a general qubit state:

$$
|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\varphi}\sin(\theta/2)|1\rangle
$$

we get:

$$
\begin{align*}
Z|\psi\rangle &= \cos(\theta/2)|0\rangle - e^{i\varphi}\sin(\theta/2)|1\rangle \\
 &= \cos(\theta/2)|0\rangle + e^{i(\varphi + \pi)}\sin(\theta/2)|1\rangle
\end{align*}
$$

In terms of spherical coordinates, $Z$ transforms the coordinates
$(\theta, \varphi)$ to $(\theta, \varphi + \pi)$. On the Bloch sphere, adding
$\pi$ to the $\varphi$ coordinate corresponds to rotation by $\pi$ radians
around the z-axis.

We can easily generalize this example to unitary matrices of the form

$$
Z_\alpha = \left[\begin{matrix} 1 & 0 \\ 0 & e^{i\alpha} \end{matrix}\right]
$$

Clearly

$$
Z_\alpha |\psi\rangle =
 \cos(\theta/2)|0\rangle + e^{i(\varphi + \alpha)}\sin(\theta/2)|1\rangle
$$

and so multiplication by $Z_\alpha$ corresponds to rotating the Bloch sphere by
$\alpha$ radians around the z-axis (see for reference
[axis-angle representation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)).

In order to formalize this correspondence, we'll define the function

$$
F : \mathrm{U}(2) \rightarrow \mathrm{Aut}(\mathbb{R}^3)
$$

which sends a unitary matrix $U$ to the corresponding automorphism of the Bloch
sphere. In other words, $F$ is defined to satisfy

$$
\mathrm{Bloch}(U|\psi\rangle) = F(U)\cdot\mathrm{Bloch}(|\psi\rangle)
$$

for any qubit state $|\psi\rangle \in \mathbb{C}^2$ and unitary
$U\in\mathrm{U}(2)$.

We will also use the notation $\mathrm{Rot}_{\mathbf{n}}(\alpha)$ to denote
[rotation](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) of
$\mathbb{R}^3$ by $\alpha$ radians around the axis $\mathbf{n}\in\mathbb{R}^3$.

Using this notation, we can restate our above observations about $Z_\alpha$ more
compactly as:

$$
F(Z_\alpha) = \mathrm{Rot}_{\mathbf{z}}(\alpha)
$$

where $\mathbf{z} = (0, 0, 1)$ denotes the z-axis.

The goal of this post is to prove the following generalization to arbitrary
unitary matrices:

{: #thm:bloch-rotation }

> **Theorem (Bloch Rotation).** Let $U\in\mathrm{U}(2)$ be a unitary matrix with
> eigenvalues $\lambda_1,\lambda_2\in\mathbb{C}$ and corresponding eigenvectors
> $|\psi_1\rangle,|\psi_2\rangle\in\mathbb{C}^2$. Then
>
> $$
> F(U) = \mathrm{Rot}_{\mathbf{n}}(\alpha)
> $$
>
> Where $\mathbf{n} = \mathrm{Bloch}(|\psi_1\rangle)$ and $\alpha$ is the angle
> satisfying $\lambda_2/\lambda_1 = e^{i\alpha}$.

To see how this works, let's apply the theorem to the unitary matrix $Z_\alpha$
from the example above. The eigenvalues of $Z_\alpha$ are $\lambda_1 = 1$ and
$\lambda_2=e^{i\alpha}$ with eigenvectors $|\psi_1\rangle=|0\rangle$ and
$\psi_2\rangle=|1\rangle$.

According to the [theorem](#thm:bloch-rotation), the axis of rotation is:

$$
\mathbf{n} = \mathrm{Bloch}(|\psi_1\rangle)
= \mathrm{Bloch}(|0\rangle) = (0, 0, 1)
$$

which is indeed equal to the z-axis $\mathbf{z}$. To find the angle of rotation,
we can compute:

$$
\lambda_2 / \lambda_1 = e^{i\alpha} / 1 = e^{i\alpha}
$$

which by the theorem implies that the angle of rotation is equal to $\alpha$ as
we found above.

Note that it is not obvious from the definition of the Bloch projection that
$F(U)$ is even a rotation at all. We can think of the
[Bloch Rotation](#thm:bloch-rotation) theorem as consisting of two parts:

1. A unitary transformation $U$ of qubit states corresponds to some rotation
   $F(U)$ of the Bloch sphere.
2. A formula relating the angle and axis of the rotation $F(U)$ to the
   eigenvalues and eigenvectors of $U$.

In the next section we will prove the angle and axis formula under the
assumption that $F(U)$ is indeed a rotation. In the following sections we'll
introduce an alternative formulation of the Bloch projection and use it to prove
that $F(U)$ is always a rotation.

Finally, in the last section we'll use the theorem to prove the standard
[rotation formula](https://en.wikipedia.org/wiki/Bloch_sphere#Rotations_about_a_general_axis)
relating the axis of rotation to the
[Pauli Matrices](https://en.wikipedia.org/wiki/Pauli_matrices).

## Proof Of The Bloch Rotation Theorem

In this section we will prove the [Bloch Rotation](#thm:bloch-rotation) theorem
under the assumption that $F(U)$ is always a rotation of $\mathbb{R}^3$. The set
of rotations of $\mathbb{R}^3$ forms a group called the
[Special Orthogonal Group](https://en.wikipedia.org/wiki/3D_rotation_group)
denoted $\mathrm{SO}(3)$. So we can rephrase our assumption on $F$ as saying
that $F$ is a function from the group of unitary matrices $\mathrm{U}(2)$ to the
group of rotations $\mathrm{SO}(3)$:

$$
F : \mathrm{U}(2) \rightarrow \mathrm{SO}(3)
$$

We saw in the previous section that the [Bloch Rotation](#thm:bloch-rotation)
theorem holds for the unitary matrices

$$
Z_\alpha = \left[\begin{matrix} 1 & 0 \\ 0 & e^{i\alpha} \end{matrix}\right]
$$

Our strategy to prove the general case is to show that an arbitrary unitary
matrix $U$ can be transformed to $Z_\alpha$ using a change of coordinates. This
will allow us to deduce the general case of the theorem from the special case of
$Z_\alpha$.

We'll start by proving some simple facts about the function $F$.

{: #lem:f-composition }

> **Lemma (Composition).** Let $U$ and $V$ be unitary matrices. Then
>
> $$
> F(UV) = F(U)F(V)
> $$

_Proof._ By the definition of $F$, for any state $\|\psi\rangle$ we have:

$$
\begin{align*}
F(UV)\mathrm{Bloch}(|\psi\rangle) &= \mathrm{Bloch}(UV|\psi\rangle) \\
&= F(U)\mathrm{Bloch}(V|\psi\rangle) \\
&= F(U)F(V)\mathrm{Bloch}(|\psi\rangle)
\end{align*}
$$

Since the Bloch projection is onto, this means that

$$
F(UV)\mathbf{v} = F(U)F(V)\mathbf{v}
$$

for any vector $\mathbf{v}$ on the Bloch sphere which implies that
$F(UV)=F(U)F(V)$.

_q.e.d_

{: #lem:f-inverse }

> **Lemma (Inverse).** Let $U\in\mathrm{U}(2)$ be a unitary matrix. Then
>
> $$
> F(U^{-1}) = F(U)^{-1}
> $$

_Proof._ This follows directly from the composition lemma above and the fact
that $F$ maps the identity in $\mathrm{U}(2)$ to the identity in
$\mathrm{SO}(3)$.

_q.e.d_

{: #lem:f-scalar-multiplication }

> **Lemma (Scalar Multiplication).** Let $U\in\mathrm{U}(2)$ be a unitary matrix
> and $\lambda\in\mathbb{C}$ be a complex number with norm $1$. Then
>
> $$
> F(\lambda U) = F(U)
> $$

_Proof_. This follows immediately from the definition of $F$ and the fact that
the Bloch projection is invariant under scalar multiplication.

_q.e.d_

{: #lem:f-z-axis }

> **Lemma (Z Axis).** Let $U\in\mathrm{U}(2)$ be a unitary matrix and let
> $|\psi\rangle\in\mathbb{C}^2$ be the first column of $U$. Then
>
> $$
> F(U)\mathbf{z} = \mathrm{Bloch}(|\psi\rangle)
> $$
>
> where $\mathbf{z}=(0,0,1)\in\mathbb{R}^3$ denotes the z-axis.

_Proof._ By the definition of $\|\psi\rangle$,

$$
U|0\rangle = |\psi\rangle
$$

Furthermore, direct calculation easily shows that

$$
\mathrm{Bloch}(|0\rangle) = \mathbf{z}
$$

The claim follows from these two observations together with the definition of
$F$:

$$
\begin{align*}
F(U)\mathbf{z} &= F(U)\mathrm{Bloch}(|0\rangle) \\
&= \mathrm{Bloch}(V|0\rangle) \\
&= \mathrm{Bloch}(|\psi\rangle)
\end{align*}
$$

_q.e.d_

We are now ready to prove the [Bloch Rotation](#thm:bloch-rotation) theorem.

Let $U\in\mathrm{U}(2)$ be a unitary matrix. By the unitary property, it has two
eigenvalues $\lambda_1,\lambda_2\in\mathbb{C}$ which both have an absolute value
of $1$
([wikipedia](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Additional_properties)).
Let $|\psi_1\rangle,|\psi_2\rangle\in\mathbb{C}^2$ be the corresponding
eigenvectors.

Let $V\in\mathbb{C}^{2\times 2}$ denote the matrix whose columns are the
eigenvectors.

By the
[eigendecomposition](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Diagonalization_and_the_eigendecomposition)
theorem, we can factor $U$ as:

$$
U = V \left[\begin{matrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{matrix}\right] V^{-1}
$$

Dividing by $\lambda_1$ and setting $\alpha$ to satisfy
$e^{i\alpha}=\lambda_2/\lambda_1$ we get:

$$
\frac{1}{\lambda_1}U = V Z_\alpha V^{-1}
$$

By the [scalar multiplication](#lem:f-scalar-multiplication),
[composition](#lem:f-composition) and [identity](#lem:f-identity) lemmas:

$$
\begin{align*}
F(U) &= F(\frac{1}{\lambda_1}U) \\
&= F(V Z_\alpha V^{-1}) \\
&= F(V) F(Z_\alpha) F(V)^{-1} \\
&= F(V)\mathrm{Rot}_\mathbf{z}(\alpha)F(V)^{-1}
\end{align*}
$$

In summary:

$$
\begin{equation}\label{eq:fu-decomp}
F(U) = F(V)\mathrm{Rot}_\mathbf{z}(\alpha)F(V)^{-1}
\end{equation}
$$

By the [z-axis](#lem:f-z-axis) lemma and the definition of $V$, $F(V)$
transforms the z-axis in $\mathbb{R}^3$ to $\mathrm{Bloch}(|\psi_1\rangle)$.
Combining this with equation \ref{eq:fu-decomp} we see that $F(U)$ first applies
the rotation $F(V)^{-1}$ which rotates $\mathrm{Bloch}(|\psi_1\rangle)$ to the
z-axis, then preforms a rotation of $\alpha$ radians around the z-axis and then
applies $F(V)$ which rotates the z-axis back to
$\mathrm{Bloch}(|\psi_1\rangle)$. This sequence of operations clearly fixes
$\mathrm{Bloch}(|\psi_1\rangle)$. It is also easy to see that it rotates the
plane orthogonal to $\mathrm{Bloch}(|\psi_1\rangle)$ by $\alpha$ radians.
Together this proves the theorem.

## Bloch From Reflections

In this section we will present an alternative formulation of the Bloch
projection. In addition to being interesting in its own right, this new
perspective will make it easy to prove that unitary transformations of qubits
always correspond to rotations of the Bloch sphere which will complete our proof
of the [Bloch Rotation](#thm:bloch-rotation) theorem.

> **Definition (Reflection).** Let $\|\psi\rangle\in\mathbb{C}^2$ be a qubit
> state. The reflection with axis $\|\psi\rangle$, denoted
> $\mathrm{Ref}(\|\psi\rangle$, is defined to be the linear transformation of
> $\mathbb{C}^2$ that fixes $\|\psi\rangle$ and scales the vector orthogonal to
> $\|\psi\rangle$ by $-1$.

More concretely, if $|\psi^\perp\rangle\in\mathbb{C}^2$ denotes a vector
orthogonal to $|\psi\rangle$ then $\mathrm{Ref}(|\psi\rangle)$ is defined to
satisfy:

$$
\begin{alignat*}{3}
\mathrm{Ref}(|\psi\rangle) &\cdot |\psi\rangle       &&=  &&|\psi\rangle \\
\mathrm{Ref}(|\psi\rangle) &\cdot |\psi^\perp\rangle &&= -&&|\psi^\perp\rangle
\end{alignat*}
$$

Clearly the eigenvalues of $\mathrm{Ref}(|\psi\rangle)$ are $1$ and $-1$ which
[implies](https://en.wikipedia.org/wiki/Hermitian_matrix#Spectral_properties)
that $\mathrm{Ref}(|\psi\rangle)$ is
[Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix). Furthermore, since
the trace of a matrix is equal to the sum of its eigenvalues, the trace of
$\mathrm{Ref}(|\psi\rangle)$ is $0$. We'll denote the vector space of
$2\times 2$ Hermitian matrices with trace $0$ by $\mathfrak{su}(2)$ and define
the _reflection function_:

$$
\begin{align*}
\mathrm{Refl} : \mathbb{C}^2 &\rightarrow \mathfrak{su}(2) \\
                |\psi\rangle &\mapsto \mathrm{Refl}(|\psi\rangle)
\end{align*}
$$

We'll now find a basis for the vector space $\mathfrak{su}(2)$. Let
$M \in  \mathfrak{su}(2)$ be a matrix with coefficients:

$$
M = \left[ \begin{matrix}a & b \\ c & d \end{matrix}\right]
$$

where $a,b,c,d \in \mathbb{C}$. The Hermitian condition means that $M=M^*$ and
so

$$
\left[ \begin{matrix}a & b \\ c & d \end{matrix}\right] =
\left[ \begin{matrix}a^* & c^* \\ b^* & d^* \end{matrix}\right]
$$

This implies that $a$ and $d$ are real numbers and that $c = b^*$.

The traceless condition implies that:

$$
\mathrm{tr}(M) = a + d = 0
$$

and so $a = -d$. Setting $a=z\in\mathbb{R}$ and $c=x+yi\in\mathbb{C}$ we can
write $M$ as:

$$
M = \left[\begin{matrix}x & z-yi \\ z+yi & -x \end{matrix}\right] =
x \left[\begin{matrix}0 & 1 \\ 1 & 0 \end{matrix}\right] +
y \left[\begin{matrix}0 & -i \\ i & 0 \end{matrix}\right] +
z \left[\begin{matrix}1 & 0 \\ 0 & -1 \end{matrix}\right]
$$

for some _real_ numbers $x,y,z\in\mathbb{R}$. The matrices

$$
\begin{align*}
X &= \left[ \begin{matrix}0 & 1 \\ 1 & 0 \end{matrix}\right] \\
Y &= \left[ \begin{matrix}0 & -i \\ i & 0 \end{matrix}\right] \\
Z &= \left[ \begin{matrix}1 & 0 \\ 0 & -1 \end{matrix}\right]
\end{align*}
$$

are called [Pauli Matrices](https://en.wikipedia.org/wiki/Pauli_matrices). It is
easy to see that all three Pauli matrices are in $\mathfrak{su}(2)$. The above
calculation shows that the Pauli matrices form a basis for $\mathfrak{su}(2)$.

We'll define the _Pauli function_

$$
\mathrm{Pauli} : \mathfrak{su}(2) \rightarrow \mathbb{R}^3
$$

to be the function that sends a matrix $M\in\mathfrak{su}(2)$ to it's
coordinates with respect to the basis of Pauli matrices
$X,Y,Z\in\mathfrak{su}(2)$.

The relationship to the Bloch projection is given by the following claim:

{: #clm:bloch-from-reflections }

> **Claim (Bloch From Reflections).** Let $|\psi\rangle\in\mathbb{C}^2$ be a
> qubit state. Then:
>
> $$
> \mathrm{Bloch}(|\psi\rangle) = \mathrm{Pauli}(\mathrm{Ref}(|\psi\rangle))
> $$

<!-- https://q.uiver.app/#q=WzAsMyxbMCwwLCJcXG1hdGhiYntDfV4yIl0sWzIsMCwiXFxtYXRoYmJ7Un1eMyJdLFsxLDEsIlxcbWF0aGZyYWt7c3V9KDIpIl0sWzAsMV0sWzAsMl0sWzIsMV1d -->
<iframe class="quiver-embed" src="https://q.uiver.app/#q=WzAsMyxbMCwwLCJcXG1hdGhiYntDfV4yIl0sWzIsMCwiXFxtYXRoYmJ7Un1eMyJdLFsxLDEsIlxcbWF0aGZyYWt7c3V9KDIpIl0sWzAsMV0sWzAsMl0sWzIsMV1d&embed" width="438" height="304" style="border-radius: 8px; border: none;"></iframe>

The [Pauli Vector](https://en.wikipedia.org/wiki/Pauli_matrices#Pauli_vectors)
is defined to be the tuple:

$$
\overrightarrow{\sigma} = \left(X,Y,Z\right)
$$

Analogously to the dot product, the product of a vector
$\mathbf{v} = (x,y,z)\in\mathbb{R}^3$ and $\overrightarrow{\sigma}$ is defined
to be:

$$
\mathbf{v} \cdot \overrightarrow{\sigma} = xX + yY + zZ
$$

By our discussion above, for any traceless Hermitian matrix
$H\in\mathfrak{su}(2)$ there exists a vector $\mathbf{v}\in\mathbb{R}^3$ such
that

$$
H = \mathbf{v} \cdot \overrightarrow{\sigma}
$$

## The Rotation Formula

We are now ready to state the correspondence between special unitary matrices
$U\in\mathrm{SU}(2)$ and rotations of the Bloch sphere.

Let $U\in\mathrm{SU}(2)$ be a special unitary matrix. By our results in section
[The Special Unitary Group](#the-special-unitary-group), there exists a
traceless Hermitian matrix $H\in\mathfrak{su}(2)$ such that

$$
U = e^{iH}
$$

As we saw in the end of the previous section, there exists a real vector
$\mathbf{v} \in \mathbb{R}^3$ such that

$$
H = \mathbf{v} \cdot \overrightarrow{\sigma}
$$

Note that we can write $\mathbf{v}$ as

$$
\mathbf{v} = \theta \mathbf{n}
$$

for where $\theta = \|\|\mathbf{v}\|\|$ and $\mathbf{n}\in\mathbb{R}^3$ is a
unit vector.

Putting this all together we can parameterize special unitary matrices $U$ by a
real number $\theta \in \mathbb{R}$ and a real unit vector
$\mathbf{n} \in \mathbb{R}^3$:

$$
U = e^{i\theta\mathbf{n}\cdot\overrightarrow{\sigma}}
$$

Finally, given a unit vector $\mathbf{n}\in\mathbb{R}^3$ and an angle
$\theta\in\mathbb{R}$, let
$R_{\mathbf{n}}(\theta)\in\mathrm{Mat}_{3\times 3}(\mathbb{R})$ denote the real
$3\times 3$
[rotation matrix](https://en.wikipedia.org/wiki/3D_rotation_group#Axis_of_rotation)
representing a 3D rotation around the axis $\mathbf{n}$ by $\theta$ radians.

We can now state the relationship between unitary transformations of quantum
states and Bloch sphere:

> Let
> $U = e^{-i\frac{\theta}{2}\mathbf{n}\cdot\overrightarrow{\sigma}} \in \mathrm{SO}(2)$
> be a special unitary matrix and let $|\psi\rangle\in\mathbb{C}^2$ be a qubit
> state. Then transforming $|\psi\rangle$ by $U$ corresponds to a rotation by
> $\theta$ radians around the axis $\mathbf{n}$ on the Bloch sphere. More
> precisely:
>
> $$
> \mathrm{Bloch}(U |\psi\rangle) = R_{\mathbf{n}}(\theta)(\mathrm{Bloch}(|\psi\rangle))
> $$

We can easily extend this formula to determine the rotation corresponding to
_any_ unitary matrix $U\in\mathrm{U}(2)$.

Recall from section [The Special Unitary Group](#the-special-unitary-group) that
the determinant of a unitary matrix has norm $1$. This means that there exists a
real number $\alpha\in\mathbb{R}$ such that

$$
\mathrm{det}(U) = e^{i\alpha}
$$

Since $U$ is a $2\times 2$ matrix, the determinant of
$V = e^{-i\frac{\alpha}{2}}U$ is equal to 1:

$$
\mathrm{det}(V) = \mathrm{det}(e^{-i\frac{\alpha}{2}}U) =
\left(e^{-i\frac{\alpha}{2}}\right)^2\mathrm{det}(U) =
e^{-i\alpha}e^{i\alpha} = 1
$$

Since $V$ is clearly unitary, this means that $V$ is a special unitary matrix.
By the definition of $V$ we have:

$$
U = e^{i\frac{\alpha}{2}}V
$$

In conclusion, for every unitary matrix $U\in\mathrm{U}(2)$ there exists a real
number $\alpha\in\mathbb{R}$ and a special unitary matrix $V\in\mathrm{SO}(2)$
such that

$$
U = e^{i\alpha}V
$$

Recall from section [Bloch Sphere](#the-bloch-sphere) that the Bloch projection
is invariant under scalar multiplications of the qubit state. This means that
the matrices $U$ and $V$ have the same effect on the Bloch sphere.

As an example, let's use the formula to determine the effect of the Pauli matrix

$$
X = \left[\begin{matrix}1 & 0 \\ 0 & -1 \end{matrix}\right]
$$
