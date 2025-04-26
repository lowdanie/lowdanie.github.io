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
