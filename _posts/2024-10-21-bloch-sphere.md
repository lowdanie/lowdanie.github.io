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

Note that this projection ignores the global phase $e^{i \varphi_0}$. The
physical motivation for this is that due to the properties of
[quantum measurement](https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics)
a global phase has no observable effect.
