---
layout: post
title: "Fully Homomorphic Encryption from Scratch"
date: 2024-01-03
mathjax: true
---

# Introduction

_Fully Homomorphic Encryption_ (FHE) is a form of encryption that makes it possible to evaluate functions on encrypted inputs without needing to decrypt them. For example, let $K$ be an encryption key and let $Enc_K(x)$ denote the encryption of the integer $x$ with key $K$. With FHE it is possible for someone who only has access to the ciphertexts $Enc_K(x)$ and $Enc_K(y)$ to evaluate the addition function and produce the ciphertext $Enc_K(x+y)$.

As a more practical example, in principal FHE makes it possible to upload an encrypted image to a cloud service, ask it to run the _encrypted_ image through a neural network classifier and get back an encrypted response. This would happen without the cloud service ever having access to the unencrypted image.

At first glance it seems like FHE should be impossible. Going back to our first example, since $Enc_K(x)$ and $Enc_K(y)$ should be indistinguishable from random bytes to someone without the key $K$, how can they perform a meaningful calculation on them to obtain $Enc_K(x+y)$?

In this post we will explore a popular FHE encryption scheme called [TFHE](https://eprint.iacr.org/2018/421.pdf) and implement it from scratch in Python. Everything in this post (and much more) can be found in that paper.

The standard cryptography exposition disclaimer applies here. Namely, that it probably is not a good idea to copy paste code from this post to secure your sensitive data. You can use the official [TFHE library](https://github.com/tfhe/tfhe) instead.

# Outline 
Encryption schemes are typically based on a fundamental mathematical problem that is assumed to be "hard". For example, the hardness of RSA is based on the assumption that is is hard to find the prime factors of large integers. Similarly, the hardness of TFHE is based on a problem called [Learning With Errors](https://en.wikipedia.org/wiki/Learning_with_errors) (LWE). In the next section we will define the LWE problem and use it to construct an encryption scheme in which it is possible to homomorphically _add_ ciphertexts but not to _multiply_ them. This type of encryption scheme is called _Partially Homomorphic_ because not all operations are supported.

Next we will consider an extension of LWE called _Ring Learning With Errors_. This is essentially a version of LWE with extra structure. We will later use this extra structure to upgrade our partially homomorphic scheme to a scheme that supports homomorphic multiplication as well as addition.

Finally, we will complete our construction of TFHE by showing how homomorphic additions and multiplications can be used to implement a homomorphic [NAND](https://en.wikipedia.org/wiki/NAND_gate) gate. I.e, that from encryptions $Enc_K(x)$ and $Enc_K(y)$ of bits $x$ and $y$ it is possible to compute $Enc_K(NAND(x, y))$ without knowing the key $K$. From there it follows that TFHE is fully homomorphic as [any boolean expression can be constructed out of NAND gates](https://en.wikipedia.org/wiki/NAND_logic).

# Learning With Errors

In this section we will define the _Learning With Errors_ (LWE) problem and use it to construct a partially homomorphic encryption scheme.

## Notation

Let $q$ be a positive integer. We will denote the integers modulo $q$ by $\mathbb{Z}_q := \mathbb{Z} / q\mathbb{Z}$. In this post, it will be convenient to set $q=2^{32}$ so that elements of $\mathbb{Z}_q$ can be represented by 32-bit integers.

## The Learning With Errors Problem

Let $\mathbf{s} \in \mathbb{Z}^n_q$ be a length $n$ vector with elements in $\mathbb{Z}_q$ (i.e int32) which is assumed to be secret. Let $\mathbf{a}_1, \ldots , \mathbf{a}_m \in \mathbb{Z}^n_q$ be random vectors and let $b_i = (\mathbf{a}_i, \mathbf{s})$ be the dot product of $\mathbf{a}_i$ with $\mathbf{s}$. As a warmup to the LWE problem we can ask:

> Given $m \geq n$ random vectors $\mathbf{a}_1, \ldots , \mathbf{a}_m$ and the dot products $b_i, \ldots , b_m$, is it possible to efficiently *learn* the secret vector $\mathbf{s}$?

It is not hard to see that the answer is "yes". Indeed, we can create an $m \times n$ matrix $A$ whose rows are the vectors $\mathbf{a}_i$ and a length $m$ vector $\mathbf{b}$ whose elements are $b_i$:

\\[
A = \begin{bmatrix} \mathbf{a}^T_1 \\\\ \vdots \\\\ \mathbf{a}^T_m \end{bmatrix}_{m \times n}
\mathbf{b} = \begin{bmatrix} b_1 \\\\ \vdots \\\\ b_m \end{bmatrix}
\\]

We can then express $\mathbf{s}$ as the solution to a linear equation:

\\[
    A \mathbf{s} = \mathbf{b}
\\]

Finally, we can use [Gaussian Elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) to solve for $\mathbf{s}$ in polynomial time. Note that the standard Gaussian elimination algorithm has to be [tweaked a bit](https://math.stackexchange.com/questions/12563/solving-systems-of-linear-equations-over-a-finite-ring) to account for the fact that $\mathbb{Z}_q$ is not a field but the general idea is the same.

We can think of the above problem as *Learning Without Errors* since we are given the exact dot products $b_i = (\mathbf{a}_i, \mathbf{s})$. It turns out that if we introduce errors by adding a bit of noise $e_i$ to each $b_i$ then learning $\mathbf{s}$ from the random vectors $A$ and the *noisy* dot products $b_i = (\mathbf{a}_i, \mathbf{s}) + e_i$ is very hard. We can now state the *Learning With Errors* problem:

> Given $m$ random vectors $\mathbf{a}_1, \ldots , \mathbf{a}_m$ and noisy dot products $b_i = (\mathbf{a}_i, \mathbf{s}) + e_i$, is it possible to efficiently learn the secret vector $\mathbf{s}$?

Note that we have not yet specified which distribution the errors $e_i \in \mathbb{Z}_q$ drawn from. In TFHE, they are sampled by first sampling a real number $-\frac{1}{2} \geq x_i < \frac{1}{2}$ from a Gaussian distribution $\mathcal{N}(0, \sigma)$, and then converting $x$ to an integer in the interval $[-\frac{q}{2}, \frac{q}{2})$ by:

\\[
    e_i = \lfloor x \cdot \frac{q}{2} \rfloor
\\]

We will denote the distribution on $\mathbb{Z}_q$ obtained in this way by $\mathcal{N}_q(0, \sigma)$.

In the next section we will see how to build a partially homomorphic encryption scheme based on the hardness of LWE.

## A LWE Based Encryption Scheme

In this section we will build a simple encryption scheme based on the LWE problem. 

### Encryption and Decryption

Following the notation in the previous section, our encryption scheme will be parameterized by a modulus $q$, a dimension $n$ and a noise level $\sigma$. Typical values would be $q=2^32$, $n=500$ and $\sigma=2^{-20}$.

The *message space* of our scheme (i.e, data type that will be encrypted) will be elements $m \in \mathbb{Z}_q$. The encryption keys will be random length $n$ binary vectors $\mathbf{s} \in \\{0, 1\\}^n \subset \mathbb{Z}^n_q$.

To encrypt a message $m \in \mathbb{Z}_q$ with a key $\mathbf{s} \in \mathbb{Z}^n_q$ we first uniformly sample a vector $\mathbf{a} \in \mathbb{Z}^n_q$ and sample a noise element $e$ from the Gaussian distribution $\mathcal{N}_q(0, \sigma)$. The encrypted message is defined to be the pair

\\[
    \mathrm{Enc}_{\mathbf{s}}(m) := (\mathbf{a}, (\mathbf{a}, \mathbf{s}) + m + e)
\\]

If we know the secret key $\mathbf{s}$, we can decrypt a ciphertext $(\mathbf{a}, b)$ by computing:

\\[
\mathrm{Dec}_{\mathbf{s}}((\mathbf{a}, b)) = (b - (\mathbf{a}, \mathbf{s}) = ((\mathbf{a}, \mathbf{s}) + m + e) - (\mathbf{a}, \mathbf{s}) = m + e
\\]

Note that this does not quite recover $m$ but rather $m+e$. For some applications such as neural networks a small amount of error may be tolerable. An alternative approach is to restrict the set of possible values of $m$ so that $m$ can be recovered from $m+e$ by rounding to the nearest allowed value.

For the purposes of this post we will only need to distinguish between 8 different messages and so we will always choose $m$ as one of the 8 multiples of $2^{29}$ modulo $2^{32}$:

IMAGE

Since the message can only have one of 8 values, we can represent it as a integer $i \in [-4, 4)$. We will call the interval $[-4, 4)$ the _restricted message space_. We will use the following *encoding* function to encode an integer $i \in [-4, 4)$ as an  element of $\mathbb{Z}_q$ before encrypting $i$:

<div style="font-size: 1.4em;">
\begin{align*}
    \mathrm{Encode}: [-4, 4) &\rightarrow \mathbb{Z}_q \\
    i &\mapsto i \cdot 2^{29} 
\end{align*} 
</div>

and the following *decoding* function to convert a 32-bit message back to an integer in $[-4, 4)$ after decryption:


<div style="font-size: 1.4em;">
\begin{align*}
    \mathrm{Decode}: \mathbb{Z}_q & \rightarrow [-4, 4) \\
    m &\mapsto \lfloor m \cdot 2^{-29} \rceil 
\end{align*} 
</div>

where $\lfloor \cdot \rceil$ denotes rounding to the nearest integer. EXPLAIN IN TERMS OF PICTURE Note that if $\vert e \vert < 2^{28}$ and $0 \leq i < 8$ then $\mathrm{Decode}(\mathrm{Encode}(i) + e) = i$. Therefore, if we encode $i$ before encrypting it and decode after decrypting we can precisely recover $i$ with no error:

\\[
\mathrm{Decode}(\mathrm{Dec}\_{\mathbf{s}}(\mathrm{Enc}\_\mathbf{s}(\mathrm{Encode}(i)))) = i
\\]

Here is an implementation of the LWE scheme we've described so far.

```python
import numpy as np
import dataclasses

# Our implementation assumes q=2^32 and so all scalars 
# will be between the minimum and maximum values of an int32.
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

@dataclasses.dataclass
class LweConfig:
    dimension: int  # The size of the secret key and ciphertext vectors.
    noise_std: float  # Standard deviation of the Gaussian noise used for encryption.

@dataclasses.dataclass
class LwePlaintext:
    """Plaintext that can be encrypted in the LWE scheme."""
    message: np.int32

@dataclasses.dataclass
class LweCiphertext:
    """The output of LWE encryption."""
    config: LweConfig
    a: np.ndarray  # shape = (config.dimension,)
    b: np.int32

@dataclasses.dataclass
class LweEncryptionKey:
    config: LweConfig
    key: np.ndarray  # shape = (config.dimension,)

def encode(i: int) -> LwePlaintext:
    """Encode an integer in [-4,4) as a plaintext."""
    return LwePlaintext(np.multiply(i, 1 << 29, dtype=np.int32))

def decode(plaintext: LwePlaintext) -> int:
    """Decode a plaintext to an integer in [-4,4)."""
    return int(np.rint(plaintext.message / (1 << 29)))

def generate_lwe_key(config: LweConfig) -> LweEncryptionKey:
    """Generate a LWE encryption key."""
    return LweEncryptionKey(
        config=config,
        key=np.random.randint(low=0, high=2,
                              size=config.dimension,
                              dtype=np.int32))

def lwe_encrypt(plaintext: LwePlaintext, key: LweEncryptionKey) -> LweCiphertext:
    """Encrypt the plaintext with the specified LWE key."""
    a = np.random.randint(
        low=INT32_MIN, high=INT32_MAX+1,
        size=key.config.dimension, dtype=np.int32)
    noise = np.int32(
        INT32_MAX  * np.random.normal(loc=0.0, scale=key.config.noise_std))

    # b = (a, key) + message + noise. All addition is done mod q=2^32.
    b = np.add(np.dot(a, key.key), plaintext.message, dtype=np.int32)
    b = np.add(b, noise, dtype=np.int32)

    return LweCiphertext(config=key.config, a=a, b=b)

def lwe_decrypt(ciphertext: LweCiphertext, key: LweEncryptionKey) -> LwePlaintext:
    """Decrypt an LWE ciphertext with the specified key."""
    # m+e = b - (a, key)
    return LwePlaintext(
        np.subtract(ciphertext.b, np.dot(ciphertext.a, key.key), 
                    dtype=np.int32))
```

Here is an example:

```python
>>> lwe_config = LweConfig(dimension=1024, noise_std=2**(-20))
>>> lwe_key = generate_lwe_key(lwe_config)

>>> # Encode the integer i=3 as a plaintext.
>>> lwe_plaintext = encode(3)
LwePlaintext(message=1610612736)
>>> # Encrypt the plaintext with the key.
>>> lwe_ciphertext = lwe_encrypt(lwe_plaintext, lwe_key)
LweCiphertext(
    config=LweConfig(dimension=1024, noise_std=9.5367431640625e-07),
    a=array([-1902953972, ..., 711394225], dtype=int32), 
    b=-109982053)
>>> # Decrypt the ciphertext. Note that the result is not exactly equal to
>>> # the plaintext as it contains some noise.
>>> lwe_decrypted = lwe_decrypt(lwe_ciphertext, lwe_key)
>>> LwePlaintext(message=1610613919)
>>> # Decode the decrypted ciphertext to an integer in the range [-4, 4).
>>> # This should give us the original integer i=3 without noise.
>>> decode(lwe_decrypted)
>>> 3
```

## LWE is Additively Homomorphic

### Motivation
In this section we will show that the LWE scheme above is *additively homomorphic*.

Before showing how this works let's take a minute to explain why this is interesting and useful. Suppose you have two secret numbers $m_1$ and $m_2$ and you want to use an untrusted server to compute their sum. Since the server is untrusted, you do not want to send $m_1$ or $m_2$. Instead you can leverage the additive homomorphic property of LWE by following these steps:

1. Generate an LWE key $\mathbf{s}$ locally.
1. Encrypt $m_1$ and $m_2$ locally to obtain $\mathrm{Enc}\_\mathbf{s}(m_1)$ and $\mathrm{Enc}\_\mathbf{s}(m_2)$.
1. Send $\mathrm{Enc}\_\mathbf{s}(m_1)$ and $\mathrm{Enc}\_\mathbf{s}(m_2)$ to the server.
1. The server uses the additive homomorphic property to compute a ciphertext $\mathrm{Enc}\_\mathbf{s}(m_1 + m_2)$. Note that the server does this *without* have access to $\mathbf{s}$, $m_1$, $m_2$ or $m_1 + m_2$.
1. The server sends $\mathrm{Enc}\_\mathbf{s}(m_1 + m_2)$ back to the client.
1. The client uses $\mathrm{s}$ to decrypt $\mathrm{Enc}\_\mathbf{s}(m_1 + m_2)$ and obtain the result $m_1 + m_2$.

### Evaluation Addition Homomorphically

Let $\mathbf{s}$ be a LWE key, let $m_1, m_2 \in \mathbb{Z}\_q$ be two messages and let  $\mathrm{Enc}\_\mathbf{s}(m_1)$ and $\mathrm{Enc}\_\mathbf{s}(m_2)$ be the corresponding ciphertexts. In order to show that LWE is additively homomorphic we must find a way to produce a ciphertext $\mathrm{Enc}\_\mathbf{s}(m_1 + m_2)$ of the sum of the messages $m_1 + m_2$ from only the ciphertexts $\mathrm{Enc}\_\mathbf{s}(m_1)$ and $\mathrm{Enc}\_\mathbf{s}(m_2)$.

As a first step, recall that by the definition of LWE encryption:

<div style="font-size: 1.4em;">
\begin{align*}
    \mathrm{Enc}_\mathbf{s}(m_1) &= (\mathbf{a_1}, \mathbf{a_1} \cdot \mathbf{s} + m_1 + e_1) \in \mathbb{Z}_q^n \times \mathbb{Z}_q \\
    \mathrm{Enc}_\mathbf{s}(m_2) &= (\mathbf{a_2}, \mathbf{a_2} \cdot \mathbf{s} + m_2 + e_2) \in \mathbb{Z}_q^n \times \mathbb{Z}_q
\end{align*} 
</div>

Where $\mathrm{a_i} \in \mathbb{Z}_q^n$ are uniformly random vectors and $e_i \in \mathbb{Z}_q$ are small errors. How can we combine these in order to get an encryption of $m_1 + m_2$? Let's try the simplest approach of just adding the terms in $\mathrm{Enc}\_\mathbf{s}(m_1)$ and $\mathrm{Enc}\_\mathbf{s}(m_2)$ element wise:

<div style="font-size: 1.4em;">
\begin{align*}
    \mathrm{Enc}_\mathbf{s}(m_1) + \mathrm{Enc}_\mathbf{s}(m_1) &= (\mathbf{a_1} + \mathbf{a_2}, (\mathbf{a_1} \cdot \mathbf{s} + m_1 + e_1) + (\mathbf{a_2} \cdot \mathbf{s} + m_2 + e_2)) \\
    &= (\mathbf{a_1} + \mathbf{a_2}, (\mathbf{a_1} + \mathbf{a_2}) \cdot \mathbf{s} + (m_1 + m_2) + (e_1 + e_2))
\end{align*} 
</div>

Let's define $\mathbf{a}\_\mathrm{sum} = \mathbf{a_1} + \mathbf{a_2}$ and $e\_\mathrm{sum} = e_1 + e_2$. We then have:

\begin{equation}\label{eq:lwe-sum}
    \mathrm{Enc}\_\mathbf{s}(m_1) + \mathrm{Enc}\_\mathbf{s}(m_1) = (\mathbf{a}\_\mathrm{sum}, \mathbf{a}\_\mathrm{sum} \cdot \mathbf{s} + (m_1 + m_2) + e\_\mathrm{sum})
\end{equation}

Note that right side of the above equation looks like an encryption of $m_1 + m_2$. Indeed, since $\mathbf{a_1}$ and $\mathbf{a_2}$ are uniformly random in $\mathbb{Z}_q^n$, $\mathbf{a}\_\mathrm{sum}$ is uniformly random as well. And since $e_1$ and $e_2$ are small errors, $e\_\mathrm{sum}$ is small as well. In summary, $\mathrm{Enc}\_\mathbf{s}(m_1) + \mathrm{Enc}\_\mathbf{s}(m_1) $ is a valid encryption of $m_1 + m_2$.

Here is an implementation:

```python
def lwe_add(
    ciphertext_left: LweCiphertext, 
    ciphertext_right: LweCiphertext) -> LweCiphertext:
    """Homomorphic addition evaluation.
    
       If ciphertext_left is an encryption of m_left and ciphertext_right is
       an encryption of m_right then return an encryption of
       m_left + m_right.
    """
    return LweCiphertext(
        ciphertext_left.config,
        np.add(ciphertext_left.a, ciphertext_right.a, dtype=np.int32),
        np.add(ciphertext_left.b, ciphertext_right.b, dtype=np.int32))

def lwe_subtract(
    ciphertext_left: LweCiphertext, 
    ciphertext_right: LweCiphertext) -> LweCiphertext:
    """Homomorphic subtraction evaluation.
    
       If ciphertext_left is an encryption of m_left and ciphertext_right is
       an encryption of m_right then return an encryption of
       m_left - m_right.
    """
    return LweCiphertext(
        ciphertext_left.config,
        np.subtract(ciphertext_left.a, ciphertext_right.a, dtype=np.int32),
        np.subtract(ciphertext_left.b, ciphertext_right.b, dtype=np.int32))
```

Here is an example:
```python
>>> lwe_config = LweConfig(dimension=1024, noise_std=2**(-20))
>>> lwe_key = generate_lwe_key(lwe_config)
>>> # Create plaintexts encoding 1 and 2.
>>> lwe_plaintext_a = encode(1)
LwePlaintext(message=536870912)
>>> lwe_plaintext_b = encode(2)
LwePlaintext(message=1073741824)
>>> # Encrypt both plaintext with the key.
>>> lwe_ciphertext_a = lwe_encrypt(lwe_plaintext_a, lwe_key)
LweCiphertext(
    config=LweConfig(dimension=1024, noise_std=9.5367431640625e-07),
    a=array([  973628608,  1574988217,  1320543876, ..., -2110200810,
       -1946314110,   176432236], dtype=int32), b=-934443317)
>>> lwe_ciphertext_b = lwe_encrypt(lwe_plaintext_b, lwe_key)
LweCiphertext(
    config=LweConfig(dimension=1024, noise_std=9.5367431640625e-07),
    a=array([1676269894,  436391436, 1556429812, ..., -943570369,  -54776011,
       1460449428], dtype=int32), b=-925793264)
>>> # Create a ciphertext of the sum.
>>> lwe_ciphertext_add = lwe_add(lwe_ciphertext_a, lwe_ciphertext_b)
LweCiphertext(
    config=LweConfig(dimension=1024, noise_std=9.5367431640625e-07),
    a=array([-1645068794,  2011379653, -1417993608, ...,  1241196117,
       -2001090121,  1636881664], dtype=int32), b=-1860236581)
>>> # Confirm that decrypting and then decoding results in the sum 1 + 2 = 3
>>> lwe_decrypted = lwe_decrypt(lwe_ciphertext_add, lwe_key)
LwePlaintext(message=1610616071)
>>> decode(lwe_decrypted)
3
```

### A Word About Noise

In the previous section we glossed over a key point related to the *noise* in the ciphertext $\mathrm{Enc}\_\mathbf{s}(m_1 + m_2)$. Recall that by \ref{eq:lwe-sum}, if the noise in $\mathrm{Enc}\_\mathbf{s}(m_1)$ is $e_1$ and the noise in $\mathrm{Enc}\_\mathbf{s}(m_2)$ is $e_2$ then after adding those two ciphertexts together the resulting ciphertext $\mathrm{Enc}\_\mathbf{s}(m_1 + m_2)$ has noise $e_1 + e_2$. Even though $e_1 + e_2$ is still "small", it is still larger than the noise in the original ciphertexts. Eventually, after a large number of additions the noise could grow to the same order of magnitude as the message itself, in which case the decryption would result in meaningless noise. To be concrete, recall that the encoding/decoding scheme from the previous section can only reliably remove noise that is less than $2^{28}$.

For this reason, there is a limit to the number of homomorphic additions we can evaluate with LWE. In the following sections we will introduce the process of *bootstrapping* which will remove this limitation.

# Ring LWE

## Introduction

The LWE scheme from the previous section was a good start, but it has a few major issues:

1. It only supports homomorphic addition.
1. It only supports a limited number of homomorphic operations (due to noise).
1. The result of encrypting a message $m \in \mathbb{Z}_q$ is a ciphertext in $\mathbb{Z}_q^n \times \mathbb{Z}_q$. For production level security $n$ is usually chosen to be around $1000$. This means that the ciphertext is around $1000$ times larger than the plaintext which can be a huge issue if the plaintext is large. For example if the plaintext is a 1MB image then after encrypting each pixel the result would be 1GB.

As we will see, *Ring LWE* (RLWE) is a variation of LWE with an additional multiplication operation. This added structure can be used to solve the ciphertext size issue, and is also an important ingredient towards the solution of the first to problems.

## Negacyclic Polynomials

The message space of the RLWE scheme will be the ring of polynomials  $\mathbb{Z}_q[x] / (x^n + 1)$. In this section we will describe the ring $\mathbb{Z}_q[x] / (x^n + 1)$ and in the next section we'll define the RLWE scheme.

For starters, $\mathbb{Z}_q[x]$ denotes the ring of polynomials with coefficients in $\mathbb{Z}_q$. As before, we will assume that $q=2^{32}$ and identify $\mathbb{Z}_q$ with signed integers. In this case $\mathbb{Z}_q[x]$ denotes the ring of polynomials integer coefficients.

An example of an element in $\mathbb{Z}_q[x]$ would be:

\\[
    f(x) = 1 + 2x + 5x^3
\\]

We call $\mathbb{Z}_q[x]$ a [Ring](https://en.wikipedia.org/wiki/Ring_(mathematics)) because we can add and multiply elements of $\mathbb{Z}_q[x]$. For example, if $f(x) = 1 + 2x + 5x^3$ and $g(x) = 1 + x$ then $f(x) + g(x) = 2 + 3x + 5x^3$ and $f(x)\cdot g(x) = 1 + 3x + 2x^2 + 5x^3 + 5x^4$.

The highest power of $x$ in a polynomial is called the *degree*. For example, the degree of $f(x)$ is $3$ and the degree of $g(x)$ is $1$. Polynomials in $\mathbb{Z}_q[x]$ can have an arbitrarily high degree.

We now turn to the ring $\mathbb{Z}_q[x] / (x^n + 1)$. The denominator $x^n+1$ plays a similar role to the denominator in the definition $\mathbb{Z}_q = \mathbb{Z} / q\mathbb{Z}$. Just as $\mathbb{Z}_q$ is defined to be the integers modulo $q$, $\mathbb{Z}_q[x] / (x^n + 1)$ is the set of polynomials "modulo" the polynomial $x^n + 1$.

What does this mean? We can represent elements of $\mathbb{Z}_q[x] / (x^n + 1)$ by polynomials whose degree is less than $n$. If we are given a polynomial with a degree larger than $n$, we can find it's representative modulo $x^n + 1$ by replacing $x^n$ with $-1$ as many times as necessary. For example, if $n=4$ then we have the following equivalences modulo $x^4 + 1$:

<div style="font-size: 1.4em;">
\begin{align*}
1 &= 1\ (\mathrm{mod}\ x^n + 1) \\
x &= x\ (\mathrm{mod}\ x^n + 1) \\
x^2 &= x^2\ (\mathrm{mod}\ x^n + 1) \\
x^3 &= x^3\ (\mathrm{mod}\ x^n + 1) \\
x^4 &= -1\ (\mathrm{mod}\ x^n + 1) \\
x^5 &= x \cdot x^4 = -x\ (\mathrm{mod}\ x^n + 1) \\
x^6 &= x^2 \cdot x^4 = -x^2\ (\mathrm{mod}\ x^n + 1) \\
x^7 &= x^3 \cdot x^4 = -x^3\ (\mathrm{mod}\ x^n + 1) \\
x^8 &= x^4 \cdot x^4 = -1 \cdot -1 = 1\ (\mathrm{mod}\ x^n + 1)
\end{align*}
</div>

Note that as the monomial pass $x^4$ they loop back to $1$ but with a negative sign. For this reason polynomials of this form are sometimes called *negacyclic*. This process is particularly useful when multiplying polynomials in $\mathbb{Z}_q[x] / (x^n + 1)$ so as to ensure that the final degree remains less than $n$. For example, let $f(x) = 1 + x^3$ and $g(x) = x$ be elements of $\mathbb{Z}_q[x] / (x^4 + 1)$. If we multiply them we get

\\[
  f(x) \cdot g(x) = x + x^4 = x - 1 = -1 + x\ (\mathrm{mod}\ x^4 + 1)
\\]

As another example that will be relevant later on, if $f(x) = 1 + x + x^2 + x^3$ and $g(x) = x^2$ then

<div style="font-size: 1.4em;">
\begin{align*}
  f(x) \cdot g(x) &= x^2 + x^3 + x^4 + x^5 \\
  &= x^2 + x^3 - 1 -x \\
  &= -1 - x  + x^2 + x^3\,  (\mathrm{mod}\,  x^4 + 1)
\end{align*}
</div>

Here is some negacyclic polynomial code that will be used later:

```python
@dataclasses.dataclass
class Polynomial:
    """A polynomial in the ring Z[x]/(x^N + 1) with int32 coefficients."""
    N: int
    # A length N array of polynomial coefficients from lowest degree to highest.
    # For example, if N=4 and f(x) = 1 + 2x + x^2 then coeff = [1, 2, 1, 0]
    coeff: np.ndarray

def polynomial_constant_multiply(c: int, p: Polynomial) -> Polynomial:
    """Multiply the polynomial by the constant c"""
    return Polynomial(coeff=np.multiply(c, p.coeff, dtype=np.int32))
    
def polynomial_multiply(p1: Polynomial, p2: Polynomial) -> Polynomial:
    """Multiply two polynomials in the ring Z[x]/(x^N + 1).
    
       We assume p1.N = p2.N.

       Note that this method is a SUBOPTIMAL way to multiply two negacyclic
       polynomials. It would be more efficient to use a negacyclic version 
       of the FFT polynomial multiplication algorithm as explained in the
       end of the post.
    """
    N = p1.N

    # Multiply and pad the result to have length 2N-1
    # Note that np.polymul expect the coefficients from highest to lowest
    # so we have to reverse coefficients before and after applying it.
    prod = np.polymul(p1.coeff[::-1], p2.coeff[::-1])[::-1]
    prod_padded = np.zeros(2*N - 1, dtype=np.int32)
    prod_padded[:len(prod)] = prod

    # Use the relation x^N = -1 to obtain a polynomial of degree N-1
    result = prod_padded[:N]
    result[:-1] -= prod_padded[N:]
    return Polynomial(N=N, coeff=result)

def polynomial_add(p1: Polynomial, p2: Polynomial) -> Polynomial:
    return Polynomial(
        N=p1.N, coeff=np.add(p1.coeff, p2.coeff, dtype=np.int32))

def polynomial_subtract(p1: Polynomial, p2: Polynomial) -> Polynomial:
    return Polynomial(
        N=p1.N, coeff=np.subtract(p1.coeff, p2.coeff, dtype=np.int32))

def zero_polynomial(N: int) -> Polynomial:
    """Build the zero polynomial in the ring Z[x]/(x^N + 1)"""
    return Polynomial(N=N, coeff=np.zeros(N, dtype=np.int32))

def build_monomial(c: int, i: int, N: int) -> Polynomial:
    """Build a monomial c*x^i in the ring Z[x]/(x^N + 1)"""
    coeff = np.zeros(N, dtype=np.int32)

    # Find k such that: 0 <= i + k*N < N
    i_mod_N = i % N
    k = (i_mod_N - i) // N

    # If k is odd then the monomial x^(i % N) picks up a negative sign since:
    # x^i = x^(i + kN - kN) = x^(-kN) * x^(i + k*N) = (x^N)^(-k) * x^(i % N) =
    #       (-1)^(-k) * x^(i % N) = (-1)^k * x^(i % N)
    sign = 1 if k % 2 == 0 else -1

    coeff[i_mod_N] = sign * c
    return Polynomial(N=N, coeff=coeff)
```

Here is an example:

```python
>>> N = 8
>>> # Build p1(x) = x + 2x + ... + 7x^7
>>> p1 = Polynomial(N=N, coeff=np.arange(N, dtype=np.int32))
Polynomial(N=8, coeff=array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32))
>>> # Build p2(x) = c*x^i = x^2
>>> p2 = build_monomial(c=1, i=2, N=N)
Polynomial(N=8, coeff=array([0, 0, 1, 0, 0, 0, 0, 0], dtype=int32))
>>> # Multiply p1 and p2
>>> q = polynomial_multiply(p1, p2)
Polynomial(N=8, coeff=array([-6, -7,  0,  1,  2,  3,  4,  5], dtype=int32))
```





