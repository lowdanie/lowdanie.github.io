---
layout: note
title: "Implicit Differentiation"
mathjax: true
---

Reference: 
[https://implicit-layers-tutorial.org/implicit_functions/](https://implicit-layers-tutorial.org/implicit_functions/)

# Implicit Differentiation

Let $f(\mathbf{a},\mathbf{z})$ be a function of the form

$$
f: \mathbb{R}^p\times\mathbb{R}^n\rightarrow\mathbb{R}^n
$$

We will think of $f$ as a family of functions from $\mathbb{R}^n$ to itself
parameterized by $\mathbf{a}\in\mathbb{R}^p$.

By the implicit function theorem, assuming that the Jacobian $\partial_\mathbf{z}f$
is non singular, the equation

$$
f(\mathbf{a}, \mathbf{z}) = 0
$$

define a function

$$
z^*: \mathbb{R}^p \rightarrow \mathbb{R}^n
$$

which sends a parameter $\mathbf{a}\in\mathbb{R}^p$ to a solution 
$z^*(\mathbf{a})\in\mathbb{R}^n$ satisfying:

$$
f(\mathbf{a}, z^*(\mathbf{a})) = 0
$$

We'd like to compute the Jacobian

$$
\partial_\mathbf{a}z^*\in\mathbb{R}^{n\times p}
$$

in terms of the partial derivatives of $f$.

Let $\mathbf{a}\_0\in\mathbb{R}^p$ be a parameter and set

$$
\mathbf{z}_0=z^*(\mathbf{a}_0)
$$

By the chain rule, the derivative of $f(\mathbf{a}, z^*(\mathbf{a}))$ at the point
$\mathbf{a}\_0$ is:

$$
\frac{\partial}{\partial\mathbf{a}} f(\mathbf{a}_0, z^*(\mathbf{a}_0))
= \partial_\mathbf{a}f(\mathbf{a}_0, \mathbf{z}_0) + 
\partial_\mathbf{z}f(\mathbf{a}_0,\mathbf{z}_0)\partial_\mathbf{a}z^*(\mathbf{a}_0)
$$

Rearranging:

$$
\partial_\mathbf{a}z^*(\mathbf{a}_0) = 
-(\partial_\mathbf{z}f(\mathbf{a}_0,\mathbf{z}_0))^{-1}
\partial_\mathbf{a}f(\mathbf{a}_0, \mathbf{z}_0)
$$

# Fixed Points

As a special case, consider the fixed point equation:

$$
f(\mathbf{a}, \mathbf{z}) = \mathbf{z}
$$

We can convert identify solutions to this equation as an implicit function by
defining:

$$
g(\mathbf{a}, \mathbf{z}) := f(\mathbf{a}, \mathbf{z}) - \mathbf{z}
$$

Solutions $(\mathbf{a}_0, \mathbf{z}_0)$ to the fixed point equation satisfy:

$$
g(\mathbf{a}_0, \mathbf{z}_0) = 0
$$

Therefore, by the implicit function theorem we can define $z^*(\mathbf{a})\in\mathbb{R}^n$
as the fixed point corresponding to $\mathbf{a}$. Furthermore, by the previous section:

$$
\begin{align*}
\partial_\mathbf{a}z^*(\mathbf{a}_0)
&= -(\partial_\mathbf{z}g(\mathbf{a}_0,\mathbf{z}_0))^{-1}
\partial_\mathbf{a}g(\mathbf{a}_0, \mathbf{z}_0) \\
&= (\mathrm{Id} - \partial_\mathbf{z}f(\mathbf{a}_0,\mathbf{z}_0))^{-1}
\partial_\mathbf{a}f(\mathbf{a}_0, \mathbf{z}_0)
\end{align*}
$$

# Jacobian Vector Products

Let $h: \mathbb{R}^n\rightarrow\mathbb{R}^m$ be a function.
The _Jacobian vector product_ (JVP) maps a pair

$$
(\mathbf{x}, \mathbf{v}) \in \mathbb{R}^n\times\mathbb{R}^n
$$

to the pair

$$
(h(\mathbf{x}), \partial h(\mathbf{x})\mathbf{v}) \in \mathbb{R}^m\times\mathbb{R}^m
$$

Forward mode auto differentiation in Jax is implemented in terms of evaluating a JVP.

Let's see how to evaluate the JVP of the fixed point function 
$z^*:\mathbb{R}^p\rightarrow\mathbb{R}^n$
from the previous section. Recall that:

$$
\partial_\mathbf{a}z^*(\mathbf{a}) = 
(\mathrm{Id} - \partial_\mathbf{z}f(\mathbf{a},\mathbf{z}))^{-1}
\partial_\mathbf{a}f(\mathbf{a}, \mathbf{z})
$$

Let $v\in\mathbb{R}^p$ be a tangent vector. We can use the JVP of $f$ with respect
to the argument $\mathbf{a}$ to evaluate

$$
\mathbf{u} := \partial_\mathbf{a}f(\mathbf{a}, \mathbf{z})\mathbf{v}
$$

We now have to compute:

$$
\mathbf{w} := (\mathrm{Id} - \partial\mathbf{z}f(\mathbf{a},\mathbf{z}))^{-1}\mathbf{u}
$$

Rearranging:

$$
(\mathrm{Id} - \partial\mathbf{z}f(\mathbf{a},\mathbf{z}))\mathbf{w} = \mathbf{u}
$$

We can solve for $\mathbf{w}$ using a linear solver such as
[GMRES](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html).
Note that GMRES solves equations of the form

$$
A\mathbf{x} = \mathbf{b}
$$

and only requires a function that can evaluate the product $A\mathbf{x}$, rather than the full
matrix $A$. In this case, we can perform this evaluation using the JVP of $f$
with respect to $\mathbf{z}$.



