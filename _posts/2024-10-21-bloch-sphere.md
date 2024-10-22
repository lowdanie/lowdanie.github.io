---
layout: post
title: "Understanding The Bloch Sphere"
date: 2024-10-21
mathjax: true
utterance-issue: 9
---

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
and therefore correspond to $2\times 2$ unitary matrices $U \in U(2)$.

The vector space $\mathbb{C}^2$ is a four dimensional real vector space, and the
constraint $|a|^2 + |b|^2 = 1$ means that the state space of a qubit is three
dimensional. It is somewhat difficult to visualize what this space looks like,
and even more difficult to visualize the effect of a unitary transformation $U$.

The _Bloch Sphere_ is an alternative representation of the state space as points
on a two dimensional 
