---
layout: post
title: "Point Cloud Alignment"
date: 2019-05-14
mathjax: true
---

# Introduction

Many robotics problems can be framed as trying to find a rotation matrix which best explains a collection of data. One example of this is the _Point Cloud Alignment_ problem in which we have two 3D point clouds and we would like to rotate the first one so that it matches up with the second as well as possible. Another classic example is _Simultaneous Localization and Mapping_ in which we observe the map from a sequence of poses (e.g, the measurements are taken at discrete intervals from a moving vehicle) and we want to determine both what the map looks like and how each of the poses are oriented relative to the map.

A fundamental issue that comes up when trying to find an optimal rotation matrix is that the the set of rotation matrices is not closed under addition. Namely, given two rotation matrices $R_1$ and $R_2$, their sum $R_1 + R_2$ may not be a rotation matrix. They are also not closed under multiplication by a scalar in the sense that for a given number $c \neq 1 \in \mathbb{R}$, $cR$ is not a rotation matrix. Another way of phrasing this is that _the set of rotation matrices do not form a vector space_.

This is a problem because many of the standard and theoretically sound optimization algorithms such as least squares assume that the quantity we are looking for is an element of a vector space. For instance, in any iterative approach we would typically start with an approximate solution $x$ and update it in the direction of $y$ with a correction of the form $x = x + \epsilon y$. Of course, such an expression would not make sense if the space of solutions is not closed under addition and multiplication by scalar.

As we will see later, the theory of Lie groups helps us circumvent this issue by providing us with a standard way to parameterize sets of matrices such as rotation matrices by a vector space. Not only is the parameterization easy to calculate, but it has many properties that are particularly useful in the context of optimization.

In the remainder of this post we will first give a formal definition of the point cloud alignment problem and discuss what tools would be needed to solve it. Next, we introduce the theory of lie groups and use it to derive a simple and elegant approach to point cloud alignment. It serves as a beautiful example of how thinking about a problem with the right framework can significantly simply the solution.

In a followup post we will see how the same theory can be used to solve different rotation optimization problems.

# Point Cloud Alignment

The motivating example for this post is called _Point Cloud Alignment_ (PCA) and is defined as follows. We are given a pair of point clouds $\mathbf{x}_1, \dots , \mathbf{x}_n$ and $\mathbf{y}_1, \dots , \mathbf{y}_n$ where all of the points are in $\mathbb{R}^3$. The goal is to find a rotation matrix $$R$$ that minimizes the following cost function:

\\[
J(R) = \sum\_{i=1}^{i=n} \Vert R\mathbf{x}\_i - \mathbf{y}\_i \Vert^2 =
\sum\_{i=1}^{i=n} (R\mathbf{x}\_i - \mathbf{y}\_i)^T (R\mathbf{x}\_i - \mathbf{y}\_i)
\\]

This type of problem comes up when we are trying to orient ourselves on a map, assuming we know our position (e.g via GPS). In that case, the target point cloud $\mathbf{y}_i$ represents positions of landmarks on the map around our current position, and the source point cloud $\mathbf{x}_i$ represents the positions of the landmarks as measured by our sensors.

Let's think through a straightforward but complicated way of solving this problem before considering a more elegant approach. If we didn't care about $R$ being a rotation matrix, solving for $R$ would be easy to do using a version of the ordinary least squares algorithm since each pair of points $\mathbf{x}_i$, $\mathbf{y}_i$ imposes a linear constraint on the matrix $R$. It turns out that this strategy can be extended to rotation matrices using the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm). That algorithm starts with the same linear equations and constrains the matrix to be a rotation matrix with Lagrange multipliers. However, we are studying PCA in order to gain intuition for how to solve more complex optimization problems involving rotation matrices and even other types of matrices such as pose matrices so we will not spend time on this particular algorithm.

To solve for a rotation matrix, we could start by  noting that since the space of rotations is 3 dimensional, we can parameterize rotations by vectors $\boldsymbol{\theta} \in \mathbb{R}^3$. There are many ways to do this, but one standard way is to think of the three scalars in $\boldsymbol{\theta}$ as rotations around each of the three axes. Then,

\\[
R(\boldsymbol{\theta}) := R_3(\theta_3)R_2(\theta_2)R_1(\theta_1)
\\]

Where $R_i(\theta)$ represents a rotation by $\theta$ radians around the $i$-th axis. We can now rewrite our cost function in terms of $\boldsymbol{\theta}$ as:

\\[
J(\boldsymbol{\theta}) =
\sum_{i=1}^{i=n} (R(\boldsymbol{\theta})\mathbf{x}_i -\mathbf{y}_i)^T(R(\boldsymbol{\theta})\mathbf{x}_i - \mathbf{y}_i)
\\]

The issue now is that the error term 

\\[
E_i(\boldsymbol{\theta}) := R(\boldsymbol{\theta})\mathbf{x}_i - \mathbf{y}_i
\\]

 is highly nonlinear in $\boldsymbol{\theta}$ due to the fact the the conversion from an angle to a rotation matrix involves multiple trigonometric functions. A standard way to deal with non linear error terms is by an iterative method such as _Gauss-Newton_. In this method, we assume that at each step we have an approximate solution $$R_\mathrm{op} = R(\boldsymbol{\theta}_\mathrm{op})$$ that we are operating on.
 To obtain a solution with lower cost, we first linearize each $E_i({\theta})$ by taking the first two terms of its Taylor series around $R_\mathrm{op}$:

\\[
E_i(\boldsymbol{\theta}\_\mathrm{op} + \delta\boldsymbol{\theta}) \approx E_i(R_\mathrm{op}) +
\nabla_{\boldsymbol{\theta}} E_i\vert\_{\boldsymbol{\theta}\_\mathrm{op}} \delta \boldsymbol{\theta}
\\]

We can now use ordinary least squares to solve for $$\delta\boldsymbol{\theta}$$, apply the update

\\[
\boldsymbol{\theta}\_\mathrm{op} = \boldsymbol{\theta}\_\mathrm{op} + \delta \boldsymbol{\theta}
\\]
, and iterate until convergence.

This strategy could work, but it requires the calculation of $\nabla_{\boldsymbol{\theta}} E_i$ which in turn involves calculating derivatives of $R(\boldsymbol{\theta})$. Since $R(\boldsymbol{\theta})$ is the product of three matrices, each of which has many trigonometric terms, calculating this derivative would be quite a mess. Furthermore, doing so would not generalize to other types of matrices such as 3d affine transformations (e.g, with the standard representation as a 4x4 matrix containing a rotation matrix and a translation vector).

In order to solve the problem more cleanly, we would like a different parameterization $R(\boldsymbol{\theta})$ of rotation matrices in terms of vectors $\boldsymbol{\theta} \in \mathbb{R}^3$ that satisfies the following two properties:

1. $R(\boldsymbol{\theta})$ is easy to differentiate as a function of the vector $\boldsymbol{\theta} \in \mathbb{R}^3$.
This requirement is helpful because once we apply the parameterization to obtain a cost function over the vector space $\mathbb{R}^3$, we want to be able to solve it by searching for local minima using gradients.

2. There should be an easy way of computing matrix multiplication in terms of the parameter $\boldsymbol{\theta}$.
Concretely, we want to be able to easily answer questions such as the following: Given a rotation matrix $R$ and the vector $\boldsymbol{\theta}_1$, find the vector $\boldsymbol{\theta}_2$ satisfying $R(\boldsymbol{\theta}_1) \cdot R = R \cdot R(\boldsymbol{\theta}_2)$.
Being able to solve for this type of equality will allow us to simplify expressions involving the parameterization $R(\boldsymbol{\theta})$.

In the following sections we will develop machinery that will lead to a parameterization of rotation matrices satisfying these properties. These techniques are quite general and can be applied to many different types of transformations and optimization functions. We will conclude by showing how this parameterization leads to a simple solution of the PCA problem.

The next two sections are a bit long, but hopefully the payoff will become clear when we we revisit PCA problem and find ourselves able to solve it with just a few lines of easy algebra. If you are skeptical, I recommend scrolling to the end where we present the final algorithm and noting the absence of any complicated equations.

# Matrix Lie Groups

In this section we'll take a step back and consider the group of $n \times n$ invertible matrices which is also known as the _general linear group_ and is denoted by $\mathrm{GL}(n)$. We call it a _group_ because there is a well defined way of multiplying two invertible matrices $A$ and $B$ to produce a third invertible matrix $C = AB$, and because every element $A \in \mathrm{GL}(n)$ has an inverse. The identity element of $\mathrm{GL}(n)$ is the $n \times n$ identity matrix which we will denote by $I_n$. Another feature of $\mathrm{GL}(n)$ is that it has the structure of a _differentiable manifold_. We do not have time to explain what that means in this post, but roughly speaking it means that there is a well defined way of doing things like taking the derivative of a function

\\[
f: \mathrm{GL(n)} \rightarrow \mathbb{R}
\\]

A group that is also a differentiable manifold is called a _Lie group_ and $\mathrm{GL}(n)$ is an important example. In fact, some of the most common types lie groups are subgroups of $\mathrm{GL}(n)$ and we call this family of lie groups _matrix Lie groups_. As we will see in the next section, the group of rotation matrices is an example of a matrix Lie group.

One fundamental property a Lie group is that we can parameterize it's elements with a vector space known as the _Lie algebra_. This allows us to reframe questions involving the group as linear algebra questions about a vector space which are typically easier to solve.

To see how this works in the case of $\mathrm{GL}(n)$, we must introduce the concept of the _matrix exponential_. To set the scene, let $\mathrm{M}(n)$ denote the vector space of all $n \times n$ matrices (not just the invertible ones). We call it a vector space because adding two elements in $\mathrm{M}(n)$ gives an element in $\mathrm{M}(n)$, as does multiplying by a scalar. The matrix exponential is very similar the usual exponential function $e^x: \mathbb{R} \rightarrow \mathbb{R}$ except that it can be applied to any $n \times n$ matrix

\\[
\mathrm{exp} : \mathrm{M}(n) \rightarrow \mathrm{M}(n)
\\]

The definition of $\mathrm{exp}$ is surprisingly simple. We just take the usual Taylor series defining the standard exponent function, and plug in matrices rather than real numbers:

\\[
\mathrm{exp}(M) := I + M + \frac{1}{2!}M^2 + \frac{1}{3!}M^3 \dots
\\]

Just like the exponent of a real number is always positive, the exponent of a matrix (even a non invertible one) is always an invertible matrix since for any matrix $M$, it is easy to show that the inverse of $\mathrm{exp}(M)$ is $\mathrm{exp}(-M)$:

\\[
\mathrm{exp}(M) \cdot \mathrm{exp}(-M) = \mathrm{exp}(M - M) = \mathrm{exp}(0_n) = I
\\]

where $0_n$ is the $n\times n$ zero matrix. Because of this, the matrix exponential defines a function from the vector space $\mathrm{M}(n)$ to the Lie group of invertible matrices $\mathrm{GL}(n)$:

\\[
\mathrm{exp} : \mathrm{M}(n) \rightarrow \mathrm{GL}(n)
\\]

This motivates us to define the Lie algebra of $\mathrm{GL}(n)$, denoted by $\mathfrak{gl}(n)$ as $\mathfrak{gl}(n) = \mathrm{M}(n)$. By definition, the matrix exponential can now be viewed as a function from the Lie algebra to the Lie group:

\\[
\mathrm{exp}: \mathfrak{gl}(n) \rightarrow \mathrm{GL}(n)
\\]

This is essentially all we need to know about $\mathrm{GL}(n)$ for the purposes of this post, and we will end this section by calling to attention a couple of key properties of the matrix exponential.

The first and easiest fact is that $$\mathrm{exp}(0_n) = I_n$$ as can be seen from plugging the zero matrix in to the Taylor series definition of $\mathrm{exp}(M)$. This is reminiscent of the standard identity $e^0 = 1$. The key takeaway from this fact is that the matrix exponential provides a parameterization of a _neighborhood_ of the identity matrix. That is to say, the matrix exponential sends the zero matrix $0_n\in \mathrm{M}(n)$ to the identity matrix, and since it's continuous, it therefore sends matrices that are close to $0_n$ to invertible matrices that are close to $I_n$. This means that if we want to construct an invertible matrix that is close to $I_n \in \mathrm{GL}(n)$, we can exponentiate a matrix that is close to $0_n$.

Another useful property has to do with taking the derivative of exponentiation. It is not hard to show (again using the Taylor series) that for any $M \in \mathrm{M}(n)$:

\\[
\frac{d}{dt}\mathrm{exp}(tM) = M \mathrm{exp}(tM)
\\]

This is analogous to the standard equality $\frac{d}{dt}e^{ct} = ce^{ct}$. This property will be extremely useful while linearizing expressions involving the matrix exponential. 

# The Special Orthogonal Group

We now apply the discussion in the previous section to the group of rotation matrices. This group is known as the _special orthogonal group_ and is defined as

\\[
\mathrm{SO}(n) := \\{ A \in \mathrm{GL}(n) \vert AA^T = I, \mathrm{det}(A) = 1 \\} \subset \mathrm{GL}(n)
\\]

Now that we've defined the Lie group, the next step is to identify its Lie algebra which we will denote by $\mathfrak{so}(n)$. As before, $\mathfrak{so}(n)$ should be a vector space which can be mapped to a neighborhood of the identity matrix $I_n \in \mathrm{SO}(n)$ via the exponential map. 

Since $\mathrm{SO}(n)$ is a subgroup of $\mathrm{GL}(n)$:

\\[
\mathrm{SO}(n) \subset \mathrm{GL}(n)
\\]

we can simply define $\mathfrak{so}(n)$ as the subspace of $\mathfrak{gl}(n)$ consisting of the elements $M \in \mathfrak{gl}(n)$ which are mapped to $\mathrm{SO}(n)$ via the matrix exponential:

\\[
\mathfrak{so}(n) := \{ M \in \mathfrak{gl}(n) | \, \mathrm{exp}(M) \in \mathrm{SO}(n) \} \subset \mathfrak{gl}(n)
\\]

The point is that now by definition, the exponential map defines a function from $\mathfrak{so}(n)$ to $\mathrm{SO}(n)$:

\\[
\mathrm{exp}: \mathfrak{so}(n) \rightarrow \mathrm{SO}(n)
\\]

This definition of $\mathfrak{so}(n)$ is not very illuminating, so we will now look for a more constructive version. As defined above, we are looking for $n \times n$ matrices $M \in \mathfrak{gl}(n) = \mathrm{M}(n)$ that satisfy the equality $\mathrm{exp}(M) \in \mathrm{SO}(n)$. By the definition of $\mathrm{SO}(n)$, this is equivalent to requiring that:

* $\mathrm{exp}(M)\mathrm{exp}(M)^T = I$

* $\mathrm{det}(\mathrm{exp}(M)) = 1$

For the first condition, we note that it is easy to show that for any matrix $X$, $\mathrm{exp}(X)^T = \mathrm{exp}(X^T)$. We have also already seen that $\mathrm{exp}(0_n) = I$. Therefore, we can simplify the first condition via:

<div style="font-size: 1.4em;">
\begin{align*}
\mathrm{exp}(M)\mathrm{exp}(M)^T = I
& \Leftrightarrow \mathrm{exp}(M + M^T) = \mathrm{exp}(0_n) \\
& \Leftrightarrow M + M^T = 0_n \\
& \Leftrightarrow M = -M^T
\end{align*}
</div>

In other words, the first condition is equivalent to $M$ being _skew-symmetric_.

To analyze the second condition we recall the following helpful identity:

\\[
\mathrm{det}(\mathrm{exp}(M)) = \mathrm{exp}(\mathrm{tr}(X))
\\]

With this in hand, it is easy to see that the second condition is equivalent to $\mathrm{tr}(M) = 0$. However, this holds for _any_ skew-symmetric matrix so the second condition turns out to be redundant.

In summary, we have shown that $\mathfrak{so}(n)$ is the vector space of skew-symmetric $n \times n$ matrices: 

\\[
\mathfrak{so}(n) = \{ M \in M(n) | M = -M^T \}
\\]

Note that this in indeed a vector space since the sum of two skew-symmetric matrices is still skew-symmetric. Furthermore, we can map this vector space to a neighborhood of the identity $I_n \in \mathrm{SO}(n)$ via the exponential map 

\\[
\mathrm{exp}: \mathfrak{so}(n) \rightarrow \mathrm{SO}(n)
\\]

# The Group $\mathrm{SO}(3)$

We now specialize further to the case of $n=3$ since after all we are interested in rotations of 3 dimensional space. In this case, $\mathrm{SO}(3)$ is the group of 3d rotation matrices, and $\mathfrak{so}(3)$ is the vector space of $3 \times 3$ skew-symmetric matrices.  

It will be helpful in the calculations that follow to obtain a concrete basis for the vector space $\mathfrak{so}(3)$. We start by noticing that every $3 \times 3$ skew-symmetric matrix $M$ can be written as:

\\[
M = \begin{bmatrix} 0 & \theta_1 & \theta_2 \\\\ -\theta_1 & 0 & \theta_3 \\\\ -\theta_2 & -\theta_3 & 0 \end{bmatrix}
\\]

This means that each skew-symmetric matrix can be identified with a column vector $(\theta_1, \theta_2, \theta_3)^T \in \mathbb{R}^3$. We will use the following standard notation called the _skew symmetrization _to relate a column vector $(\theta_1, \theta_2, \theta_3)^T \in \mathbb{R}^3$ to its corresponding skew-symmetric matrix

\\[
{\begin{bmatrix} \theta_1 \\\\ \theta_2 \\\\ \theta_3 \end{bmatrix}}_{\times} :=
\begin{bmatrix} 0 & \theta_1 & \theta_2 \\\\ -\theta_1 & 0 & \theta_3 \\\\ -\theta_2 & -\theta_3 & 0 \end{bmatrix}
\\]

In particular, we can write every antisymmetric matrix $M$ as a linear combination of the matrices

<div>
\begin{equation*}
(\mathbf{e}\_1)\_\times =
\begin{bmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 0
\end{bmatrix}, \,
(\mathbf{e}\_2)\_\times =
\begin{bmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ -1 & 0 & 0
\end{bmatrix}, \,
(\mathbf{e}\_3)\_\times =
\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0
\end{bmatrix}
\end{equation*}
</div>

Where $\mathbf{e}_i$ is the i-th standard basis vector of $\mathbb{R}^3$. Concretely, we have:

\\[
\begin{bmatrix} 0 & \theta_1 & \theta_2 \\\\ -\theta_1 & 0 & \theta_3 \\\\ -\theta_2 & -\theta_3 & 0 \end{bmatrix} =
\theta\_1 (\mathbf{e}\_1)\_\times + \theta\_2 (\mathbf{e}\_2)\_\times + \theta_3 (\mathbf{e}\_3)\_\times
\\]

In conclusion, we've shown that $\mathfrak{so}(3)$ is a 3 dimensional vector space with a basis given by
$\{( \mathbf{e}\_1)\_\times, (\mathbf{e}\_2)\_\times, (\mathbf{e}\_3)\_\times \}$. 

In light of the correspondence between elements in $\mathfrak{so}(3)$ and vectors $\boldsymbol{\theta} = (\theta_1, \theta_2, \theta_3)^T \in \mathbb{R}^3$, we will sometimes be a bit lazy and describe elements of $\mathfrak{so}(3)$ as column vectors. Furthermore, we will overload the exponentiation function and define it on column vectors $\boldsymbol{\theta}$ via

\\[
\mathrm{exp}(\boldsymbol{\theta}) := \mathrm{exp}(\boldsymbol{\theta}_\times)
\\]

A fascinating fact which we unfortunately do not have time to discuss is that $$\mathrm{exp}(\boldsymbol{\theta})$$ turns out to be the rotation matrix corresponding to the rotation around the unit vector $$\frac{\boldsymbol{\theta}}{\Vert\boldsymbol{\theta}\Vert}$$ by $$\Vert\boldsymbol{\theta}\Vert$$ radians!

In summary, we have seen that the lie algebra $\mathfrak{so}(3)$ of the group of rotation matrices $\mathrm{SO}(3)$ is equal to the vector space of skew-symmetric matrices. In addition, elements of $\mathfrak{so}(3)$ may be represented as column vectors $\boldsymbol{\theta} \in \mathbb{R}^3$. Finally, we can relate elements of this vector space to rotation matrices via the exponential map

\\[
\mathrm{exp}(\boldsymbol{\theta}) = \mathrm{exp}(\boldsymbol{\theta}_\times) \in \mathrm{SO}(3)
\\]

This relationship between a concrete vector space $\mathfrak{so}(3) \simeq \mathbb{R}^3$ and rotation matrices is what will allow us to convert optimization problems involving matrices into simpler linear algebra problems.

# Back to PCA

In this section we will put everything together an use our understanding of the lie algebra $\mathfrak{so}(3)$ to a develop a simple algorithm for point cloud alignment.

Recall that the goal of PCA is to find a rotation matrix $R \in \mathrm{SO}(3)$ that minimizes the cost

\\[
J(R) = \sum\_{i=1}^{i=n} \Vert R\mathbf{x}\_i - \mathbf{y}\_i \Vert^2 =
\sum_\{i=1}^{i=n} E\_i(R)^T E\_i(R)
\\]

where we are defining the error term $E_i(R) = R\mathbf{x}_i - \mathbf{y}_i$.

We will solve this problem with an iterative approach where we assume that at each step we have an approximate solution $R_\mathrm{op}$ which we are trying to improve. Since we can identify rotation matrices that are close to the identity matrix by elements of $\mathfrak{so}(3)$, we can parameterize rotation matrices near $R_\mathrm{op}$ by

\\[
R(\boldsymbol{\theta}) := \mathrm{exp}(\boldsymbol{\theta})R_\mathrm{op}
\\]

We can now define our error term $E_i$ as a function of  $\mathfrak{so}(3) \simeq \mathbb{R}^3$

<div style="font-size: 1.4em;">
\begin{align*}
E_i : & \mathbb{R}^3 \rightarrow \mathbb{R}^3 \\
& \boldsymbol{\theta} \mapsto \mathrm{exp}(\boldsymbol{\theta})R_\mathrm{op}\mathbf{x}_i - \mathbf{y}_i
\end{align*}
</div>

Note that $E_i(0) = R_\mathrm{op}\mathbf{x}_i - \mathbf{y}_i$. Our cost function now becomes

\\[
J(\boldsymbol{\theta}) = \sum_{i=1}^{i=n} E_i(\boldsymbol{\theta})^T E_i(\boldsymbol{\theta})
\\]

In order reduce this to a least squares problem, we will linearize the function $E_i : \mathbb{R}^3 \rightarrow \mathbb{R}^3$ by taking the first two terms of its Tayor series expansion around zero:

\\[
E_i(\boldsymbol{\theta}) \approx E_i(\mathbf{0}) +
\nabla_{\boldsymbol{\theta}} E_i|_{\mathbf{0}} \boldsymbol{\theta}
\\]

All that remains is to compute the $3 \times 3$ gradient matrix $\nabla_{\boldsymbol{\theta}} E_i$. Recall that by definition, the columns of this matrix are given by:

\\[
\nabla_{\boldsymbol{\theta}} E_i := \left[
\left. \frac{\partial}{\partial \theta_1}E_i \right\rvert
\left. \frac{\partial}{\partial \theta_2}E_i \right\rvert
\frac{\partial}{\partial \theta_3}E_i \right]
\\]

We will calculate the first column and the second two will be analogous. The calculation goes as follows

<div style="font-size: 1.4em;">
\begin{align*}
\left. \frac{\partial}{\partial \theta_1}E_i \right\rvert_\mathbf{0} &=
\left. \frac{\partial}{\partial \theta_1}\mathrm{exp}(\boldsymbol{\theta}) \right\rvert_\mathbf{0} R_\mathrm{op}\mathbf{x}_i \\
&= \left. \frac{\partial}{\partial \theta_1}\mathrm{exp}(\theta_1 (\mathbf{e}_1)_\times + \theta_2 (\mathbf{e}_2)_\times + \theta_3 (\mathbf{e}_3)_\times) \right\rvert_\mathbf{0} R_\mathrm{op}\mathbf{x}_i \\
&= (\mathbf{e}_1)_\times \cdot R_\mathrm{op}\mathbf{x}_i =
-(R_\mathrm{op}\mathbf{x}_i)_\times \mathbf{e}_1
\end{align*}
</div>

For the second equality we used the identity

\\[
\boldsymbol{\theta}\_\times = \theta_1 (\mathbf{e}\_1)\_\times + \theta_2 (\mathbf{e}\_2)\_\times + \theta_3 (\mathbf{e}\_3)\_\times
\\]

that we saw above. 

The third equality is easy to show and is analogous to the following identity that we mentioned earlier:

\\[
\frac{d}{dt}\mathrm{exp}(tM) = M \mathrm{exp}(tM)
\\]

The fourth equality comes from a general fact about the skew-symmetrization operator:

\\[
\mathbf{v}\_\times \cdot \mathbf{w} = -\mathbf{w}\_\times \cdot \mathbf{v}
\\]

In conclusion, the first column of $\frac{\partial}{\partial \theta_1}E_i\vert_\mathbf{0}$ is simply the first column of $-(R_\mathrm{op}\mathbf{x}_i)_\times$. Since the analogous result holds for the rest of the columns, we get the simple expression

\\[
\left. \frac{\partial}{\partial \theta\_1}E\_i \right\rvert\_\mathbf{0} =
-(R\_\mathrm{op}\mathbf{x}\_i)\_\times
\\]

Putting it all together, we can use the following algorithm to optimize for $R$, given an initial approximation $R_\mathrm{op}$.

1. Compute an approximate linear error term: $$E_i(\boldsymbol{\theta}) = (R_\mathrm{op}\mathbf{x}_i - \mathbf{y}_i) -(R_\mathrm{op}\mathbf{x}_i)_\times \boldsymbol{\theta}$$

2. Find the value of $$\boldsymbol{\theta} \in \mathbb{R}^3$$ that minimizes $$J(\boldsymbol{\theta}) = \sum_{i=1}^{i=n} E_i(\boldsymbol{\theta})^T E_i(\boldsymbol{\theta})$$ using least squares. Denote the solution by $$\boldsymbol{\theta}_\mathrm{min}$$.

3. Update our approximate solution to $$R_\mathrm{op} = \mathrm{exp}(\boldsymbol{\theta}_\mathrm{min})R_\mathrm{op}$$.

4. If the approximate solution has changed by more than some convergence threshold, go back to step 1. Otherwise, return $R_\mathrm{op}$.