---
layout: post
title: "Fully Homomorphic Encryption from Scratch - DRAFT"
date: 2024-01-03
mathjax: true
---

# Introduction

_Fully Homomorphic Encryption_ (FHE) is a form of encryption that makes it
possible to evaluate functions on encrypted inputs without needing to decrypt
them. For example, suppose you want to use a neural network running on an
untrusted server to determine whether your image contains a cat. Since the
server is untrusted, you do not want it to see your image. With FHE you could
encrypt the image with your secret key and upload the encrypted image to the
server. The server could then apply its neural network to the encrypted image
and produce an encrypted bit. You could then download the encrypted bit and
decrypt it with your secret key to obtain the result.

In other words, FHE makes it possible to leverage the computational resources of
an untrusted server without having to send it any unencrypted data.

At first glance it seems like FHE should be impossible. Encrypted data should be
indistinguishable from random bytes to a person without access to the encryption
key. So how could such a person perform a meaningful calculation on the
encrypted data?

In this post we will explore a popular FHE encryption scheme called
[TFHE](https://eprint.iacr.org/2018/421.pdf) and implement it from scratch in
Python. The git repo accompanying this post can be found here:
[https://github.com/lowdanie/tfhe](https://github.com/lowdanie/tfhe)

The standard cryptography exposition disclaimer applies. Namely, that it
probably is not a good idea to copy paste code from this post to secure your
sensitive data. You can use the
[official TFHE library](https://github.com/tfhe/tfhe) instead.

# A Fully Homomorphic NAND Gate

Any boolean function can be implemented with a combination of
[NAND](https://en.wikipedia.org/wiki/NAND_gate) gates. Therefore, in this post
we'll focus on implementing a homomorphic version of the NAND gate which can be
evaluated on encrypted inputs.

Specifically, we'll implement `generate_key`, `encrypt` and `decrypt` functions:

```python
def generate_key() -> EncryptionKey:
    pass

def encrypt(plaintext: Plaintext, key: EncryptionKey) -> Ciphertext:
    pass

def decrypt(ciphertext: Ciphertext, key: EncryptionKey) -> Plaintext:
    pass
```

and a homomorphic NAND gate:

```python
def homomorphic_nand(
    ciphertext_left: Ciphertext,
    ciphertext_right: Ciphertext,
) -> Ciphertext:
    """Homomorphically evaluate the NAND function."""
    pass
```

The homomorphic property will allow us to perform the following type of
computation:

```python
# Client side: Create an encryption key and encrypt two bits.
client_key = generate_key()

b_0 = True
b_1 = False

ciphertext_0 = encrypt(b_0, client_key)
ciphertext_1 = encrypt(b_1, client_key)

# The client sends the ciphertexts to the server.
# Server side: Compute an encryption of NAND(b_0, b_1) from
# the two ciphertexts:
ciphertext_nand = homomorphic_nand(ciphertext_0, ciphertext_1)

# The server sends ciphertext_nand back to the client.
# Client side: Decrypt ciphertext_nand.
# The result should be equal to NAND(True, False) = True
result = decrypt(ciphertext_nand, client_key)
assert result
```

Importantly, the server never receives the `client_key` and so it is not able to
read the contents of `ciphertext_left` or `ciphertext_right`. Nevertheless, it
can run the `homomorphic_nand` function on the ciphertexts to produce
`ciphertext_nand` which encrypts `NAND(b_0, b_1)`. Without `client_key`, the
server also cannot read the contents of `ciphertext_nand`. All if can do is send
`ciphertext_nand` back to the client whose can use `client_key` to decrypt it
and reveal the result.

Of course, all of this complexity is unnecessary if the client is only
interested in evaluating a single NAND gate. The client would be better off just
evaluating the NAND function locally not involving a server at all.

Homomorphic encryption becomes more useful when the client wants to evaluate a
complex circuit such as a neural network which is composed of many billions of
NAND gates. In that case the client may not have the compute or the permission
to run the program locally. The can offload the computation to an untrusted
using the approach in the code snippet above, where the `homomorphic_nand`
server side evaluation is used to evaluate all of the NAND gates in the complex
circuit.

The goal of this post is to implement `generate_key`, `encrypt`, `decrypt` and
`homomorphic_nand`.

# Learning With Errors

In this section we wil implement `generate_key`, `encrypt` and `decrypt`. This
combination of functions is typically called an _Encryption Scheme_.

Encryption schemes are usually based on a fundamental mathematical problem that
is assumed to be "hard". For example,
[RSA](<https://en.wikipedia.org/wiki/RSA_(cryptosystem)>) is based on the
assumption that is is hard to find the prime factors of large integers.
Similarly, TFHE is based on a problem called
[Learning With Errors](https://en.wikipedia.org/wiki/Learning_with_errors)
(LWE). In the next section we will define the LWE problem and use it to
construct an encryption scheme.

As we will see, in the LWE encryption scheme it is possible to homomorphically
_add_ ciphertexts but not to _multiply_ them. This type of encryption scheme is
called _Partially Homomorphic_ because not all operations are supported. Despite
this limitation, the partially homomorphic operations in the LWE scheme will be
useful stepping stones on the path towards fully homomorphic encryption.

In this section we will define the _Learning With Errors_ (LWE) problem and use
it to implement our encryption scheme as well as some partially homomorphic
operations.

## Notation

Let $q$ be a positive integer. We will denote the integers modulo $q$ by
$\mathbb{Z}_q := \mathbb{Z} / q\mathbb{Z}$. In this post, it will be convenient
to set $q=2^{32}$ so that elements of $\mathbb{Z}_q$ can be represented by the
32-bit integers $[-2^{31}, 2^{31})$. One advantage of using 32-bit integers is
that all operations are natively done modulo $2^{32}$.

Similarly, $\mathbb{Z}_8$ will denote the integers modulo $8$. We will identify
$\mathbb{Z}_8$ with the set of integers $[-4, 4)$.

## The Learning With Errors Problem

Let $\mathbf{s} \in \mathbb{Z}^n_q$ be a length $n$ vector with elements in
$\mathbb{Z}_q$ which is assumed to be secret. Let
$\mathbf{a}_1, \ldots , \mathbf{a}_m \in \mathbb{Z}^n_q$ be random vectors and
let $b_i = \mathbf{a}_i \cdot \mathbf{s}$ be the dot product of $\mathbf{a}_i$
with $\mathbf{s}$. As a warmup to the LWE problem we can ask:

> Given $m \geq n$ random vectors $\mathbf{a}\_1,\dots,\mathbf{a}\_m$ and the
> dot products $b_i, \ldots , b_m$, is it possible to efficiently _learn_ the
> secret vector $\mathbf{s}$?

It is not hard to see that the answer is "yes". Indeed, we can create an
$m \times n$ matrix $A$ whose rows are the vectors $\mathbf{a}_i$ and a length
$m$ vector $\mathbf{b}$ whose elements are $b_i$:

\\[ A = \begin{bmatrix} \mathbf{a}^T_1 \\\\ \vdots \\\\ \mathbf{a}^T_m
\end{bmatrix}_{m \times n} \mathbf{b} = \begin{bmatrix} b_1 \\\\ \vdots \\\\ b_m
\end{bmatrix} \\]

We can then express $\mathbf{s}$ as the solution to the linear equation:

\\[ A \mathbf{s} = \mathbf{b} \\]

Finally, we can use
[Gaussian Elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) to
solve for $\mathbf{s}$ in polynomial time. Note that the standard Gaussian
elimination algorithm has to be
[tweaked a bit](https://math.stackexchange.com/questions/12563/solving-systems-of-linear-equations-over-a-finite-ring)
to account for the fact that $\mathbb{Z}_q$ is not a field - but the general
idea is the same.

We can think of the above problem as _Learning Without Errors_ since we are
given the exact dot products $b_i = \mathbf{a}\_i \cdot \mathbf{s}$. It turns
out that if we introduce errors by adding a bit of noise $e_i$ to each $b_i$,
then learning $\mathbf{s}$ from the random vectors $A$ and the _noisy_ dot
products $b_i = \mathbf{a}\_i \cdot \mathbf{s} + e_i$ is very hard. We can now
state the
[Learning With Errors](https://en.wikipedia.org/wiki/Learning_with_errors)
problem:

> Given $m$ random vectors $\mathbf{a}_1, \ldots , \mathbf{a}_m$ and noisy dot
> products $b_i = \mathbf{a}_i \cdot \mathbf{s} + e_i$, is it possible to
> efficiently learn the secret vector $\mathbf{s}$?

Note that we have not yet specified which distribution the errors
$e_i \in \mathbb{Z}_q$ are drawn from. In TFHE, they are sampled by first
sampling a real number $-\frac{1}{2} \leq x_i < \frac{1}{2}$ from a Gaussian
distribution $\mathcal{N}(0, \sigma)$, and then scaling $x$ to obtain an integer
in the interval $[-\frac{q}{2}, \frac{q}{2})$:

\\[ e = \lfloor x \cdot \frac{q}{2} \rfloor \\]

We will denote the distribution on $\mathbb{Z}_q$ obtained in this way by
$\mathcal{N}_q(0, \sigma)$.

In the next section we will see how to build a partially homomorphic encryption
scheme based on the hardness of LWE.

## An LWE Based Encryption Scheme

Following the notation of the previous section, our encryption scheme will be
parameterized by a modulus $q$, a dimension $n$ and a noise level $\sigma$.
Typical values would be $q=2^{32}$, $n=500$ and $\sigma=2^{-20}$.

The valid inputs to an encryption function are known as the _message space_. The
message space of the LWE scheme is $\mathbb{Z}\_q$. We will also call valid
messages _plaintexts_. The encryption keys are random length $n$ binary vectors
$\mathbf{s} \in \\{0, 1\\}^n \subset \mathbb{Z}^n_q$.

To encrypt a message $m \in \mathbb{Z}\_q$ with a key
$\mathbf{s} \in \mathbb{Z}^n_q$ we first uniformly sample a vector
$\mathbf{a} \in \mathbb{Z}^n_q$ and sample a noise element $e$ from the Gaussian
distribution $\mathcal{N}\_q(0, \sigma)$. The encrypted message, also known as
the _ciphertext_, is defined to be the pair

\\[ \mathrm{Enc}\_{\mathbf{s}}(m) := (\mathbf{a}, \mathbf{a} \cdot \mathbf{s} +
m + e) \in \mathbb{Z}\_q^n \times \mathbb{Z}\_q \\]

If we know the secret key $\mathbf{s}$, we can decrypt a ciphertext
$(\mathbf{a}, b)$ by computing:

\\[ \mathrm{Dec}_{\mathbf{s}}((\mathbf{a}, b)) = b - \mathbf{a} \cdot \mathbf{s}
= (\mathbf{a} \cdot \mathbf{s} + m + e) - \mathbf{a} \cdot \mathbf{s} = m + e
\\]

Note that the encryption function is not deterministic. It turns out that this
is a feature and not a bug. Indeed, any
[semantically secure](https://en.wikipedia.org/wiki/Semantic_security)
encryption scheme _must_ have a non-deterministic encryption function. To see
why, suppose we use the scheme to encrypt yes/no responses to a poll. If the
encryption function was deterministic, we could tell if two people voted the
same way by comparing their encrypted responses. In a non-deterministic scheme,
this attack no longer works since even two people that voted the same way will
have different encrypted responses.

Let $m$ be an LWE plaintext. Since the encryption function is non-deterministic,
rather than saying that a ciphertext $L = (\mathbf{a}, b)$ is _the_ encryption
of $m$ we will say that it is _an_ encryption of $m$. The set of valid
encryptions of $m$ with key $\mathbf{s}$ will be denoted
$\mathrm{LWE}_{\mathbf{s}}(m)$. If $L = \mathrm{Enc}\_{\mathbf{s}}(m)$ then $L$
is an encryption of $m$ and so we can write:

\\[ L \in \mathrm{LWE}_{\mathbf{s}}(m) \\]

Here is an implementation of the LWE encryption scheme:

<div>
<a href="https://github.com/lowdanie/tfhe/blob/main/tfhe/lwe.py">tfhe/lwe.py</a>
</div>

```python
import dataclasses

import numpy as np

from tfhe import utils
@dataclasses.dataclass
class LweConfig:
    # Size of the LWE encryption key.
    dimension: int

    # Standard deviation of the encryption noise.
    noise_std: float


@dataclasses.dataclass
class LwePlaintext:
    message: np.int32


@dataclasses.dataclass
class LweCiphertext:
    config: LweConfig
    a: np.ndarray  # An int32 array of size config.dimension
    b: np.int32


@dataclasses.dataclass
class LweEncryptionKey:
    config: LweConfig
    key: np.ndarray  # An int32 array of size config.dimension


def generate_lwe_key(config: LweConfig) -> LweEncryptionKey:
    return LweEncryptionKey(
        config=config,
        key=np.random.randint(
            low=0, high=2, size=(config.dimension,), dtype=np.int32
        ),
    )


def lwe_encrypt(
    plaintext: LwePlaintext, key: LweEncryptionKey
) -> LweCiphertext:
    a = utils.uniform_sample_int32(size=key.config.dimension)
    noise = utils.gaussian_sample_int32(std=key.config.noise_std, size=None)

    # b = (a, key) + message + noise
    b = np.add(np.dot(a, key.key), plaintext.message, dtype=np.int32)
    b = np.add(b, noise, dtype=np.int32)

    return LweCiphertext(config=key.config, a=a, b=b)


def lwe_decrypt(
    ciphertext: LweCiphertext, key: LweEncryptionKey
) -> LwePlaintext:
    return LwePlaintext(
        np.subtract(ciphertext.b, np.dot(ciphertext.a, key.key), dtype=np.int32)
    )
```

where the `utils` module contains:

<div>
<a href="https://github.com/lowdanie/tfhe/blob/main/tfhe/utils.py">tfhe/utils.py</a>
</div>

```python
from typing import Optional

import numpy as np

INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


def uniform_sample_int32(size: int) -> np.ndarray:
    return np.random.randint(
        low=INT32_MIN,
        high=INT32_MAX + 1,
        size=size,
        dtype=np.int32,
    )


def gaussian_sample_int32(std: float, size: Optional[float]) -> np.ndarray:
    return np.int32(INT32_MAX * np.random.normal(loc=0.0, scale=std, size=size))
```

Throughout this post, we will used the following `LweConfig` whose parameters
are taken from the popular
[Lattice Estimator](https://github.com/malb/lattice-estimator).

<div>
<a href="https://github.com/lowdanie/tfhe/blob/main/tfhe/config.py">tfhe/config.py</a>
</div>

```python
from tfhe import lwe

LWE_CONFIG = lwe.LweConfig(dimension=1024, noise_std=2 ** (-24))
```

## Ciphertext Noise

Let $m \in \mathbb{Z}\_q$ be an LWE plaintext and let $\mathbf{s}$ be an LWE
encryption key. Let $L \in \mathrm{LWE}_{\mathbf{s}}(m)$ be an LWE encryption of
$m$. In the previous section we saw that:

\\[ \mathrm{Dec}_{\mathbf{s}}(L) = m + e \\]

where $e$ is some small noise whose magnitude depends on the LWE parameters. In
other words, when we decrypt $L$ we get the original message $m$ together with a
small error term. We will call $e$ the _noise_ in the ciphertext $L$.

As an example, we'll use the code above to encrypt the message $m = 2^{29}$ 1000
times and plot a histogram of the ciphertext noise. We'll be using the LWE
config above where the encryption noise has a standard deviation of $2^{-24}$.
Since we multiply this by $q/2 = 2^{31}$ during encryption, we expect the
ciphertext noise to have a standard deviation of roughly
$\sigma=2^{31-24} = 2^7 = 128$.

```python
# Generate an LWE key.
key = lwe.generate_lwe_key(config.LWE_CONFIG)

# This is the plaintext that we will encrypt.
plaintext = lwe.LwePlaintext(2**29)

# Encrypt the plaintext 1000 times and store the error of each ciphertext.
errors = []
for _ in range(1000):
    ciphertext = lwe.lwe_encrypt(plaintext, key)
    errors.append(lwe.lwe_decrypt(ciphertext, key).message - plaintext.message)
```

Here is a histogram of `errors`:

![LWE Ciphertext Noise](/assets/tfhe/lwe_noise_hist.png){: .center-image}

This matches our estimated standard deviation of $\sigma=128$ pretty well.

## Message Encoding

In the previous section we saw that when we decrypt a ciphertext with LWE we get
the original message plus a small error. For some applications such as neural
networks a small amount of error may be tolerable. An alternative approach is to
restrict the set of possible messages so that a message $m$ can be recovered
from $m+e$ by rounding to the nearest message in the restricted set.

For the purposes of this post we will only need to distinguish between 8
different messages and so all of our messages will be of the form
$m = i \cdot 2^{29}$ where $i \in \mathbb{Z}_8 = [-4, 4)$.

Here is a depiction of $\mathbb{Z}_q$ as a circle starting from $2^{31}$ in the
"3 o'clock" position and going counter clockwise all the way around to
$-2^{31}$. Note that $2^{31} = -2^{31}$ modulo $q=2^{32}$
[which is why](https://en.wikipedia.org/wiki/Modular_arithmetic) we are
depicting $\mathbb{Z}_q$ as a closed circle. The eight messages
$-4\cdot 2^{29}, \dots, 3\cdot 2^{29}$ are drawn as blue dots:

![Embedding of Z_8 in Z_32](/assets/tfhe/z8.png){: .center-image}

We will use the following encoding function to encode an integer
$i \in \mathbb{Z}_8$ as an LWE plaintext in $\mathbb{Z}_q$ (i.e as a 32-bit
integer):

<div>
\begin{align*}
    \mathrm{Encode}: \mathbb{Z}_8 &\rightarrow \mathbb{Z}_q \\
    i &\mapsto i \cdot 2^{29}
\end{align*}
</div>

Similarly, we will use the following decoding function to convert a 32-bit LWE
plaintext in $\mathbb{Z}_q$ back to an integer in $\mathbb{Z}_8$:

<div>
\begin{align*}
    \mathrm{Decode}: \mathbb{Z}_q & \rightarrow [-4, 4) \\
    m &\mapsto \lfloor m \cdot 2^{-29} \rceil
\end{align*}
</div>

where $\lfloor \cdot \rceil$ denotes rounding to the nearest integer. In terms
of the image above, the decoding function maps a point on the circle to the
nearest blue dot.

The green segment of the circle depicts points whose distance from the message
$m = 3 \cdot 2^{29}$ is less than $2^{28}$. Note that for the points on the
green segment, the closest blue dot is still $m = 3\cdot 2^{29}$. In general, if
$i \in \mathbb{Z}_8$ and $\vert e \vert < 2^{28}$ then

\\[ \mathrm{Decode}(\mathrm{Encode}(i) + e) = i \\]

In the previous section we saw that, using our standard LWE parameters, the
distribution of LWE ciphertext errors has a standard deviation of around $2^7$
which is significantly less than $2^{28}$. This means that if
$m = \mathrm{Encode}(i)$ and $L \in \mathrm{LWE}(m)$ is an encryption of $m$
then we can decrypt $L$ and remove the noise by decoding:

\\[
\mathrm{Decode}(\mathrm{Dec}\_{\mathbf{s}}(\mathrm{Enc}\_{\mathbf{s}}(\mathrm{Encode}(i))))
= i \\]

In summary, if we only encrypt messages that are encodings of $\mathrm{Z}_8$
then we can use the decoding function to remove the noise from our decryption
results.

Here is an implementation of the encoding and decoding functions:

<div>
<a href="https://github.com/lowdanie/tfhe/blob/main/tfhe/utils.py">tfhe/utils.py</a>
</div>

```python
def encode(i: int) -> np.int32:
    """Encode an integer in [-4, 4) as an int32"""
    return np.multiply(i, 1 << 29, dtype=np.int32)


def decode(i: np.int32) -> int:
    """Decode an int32 to an integer in the range [-4, 4) mod 8"""
    d = int(np.rint(i / (1 << 29)))
    return ((d + 4) % 8) - 4
```

<div>
<a href="https://github.com/lowdanie/tfhe/blob/main/tfhe/lwe.py">tfhe/lwe.py</a>
</div>

```python
def lwe_encode(i: int) -> LwePlaintext:
    """Encode an integer in [-4,4) as an LWE plaintext."""
    return LwePlaintext(utils.encode(i))


def lwe_decode(plaintext: LwePlaintext) -> int:
    """Decode an LWE plaintext to an integer in [-4,4) mod 8."""
    return utils.decode(plaintext.message)
```

In the following example we'll use the encoding and decoding functions to
encrypt a message and recover the exact message without noise after decryption.

```python
key = lwe.generate_lwe_key(config.LWE_CONFIG)

# Encode the number 2 as an LWE plaintext
plaintext = lwe.lwe_encode(2)

# Encrypt the plaintext
ciphertext = lwe.lwe_encrypt(plaintext, key)

# Decrypt the ciphertext. The result will contain noise.
decrypted = lwe.lwe_decrypt(ciphertext, key)

# Decode the result of the decryption. The decoded decryption
# should be exactly equal to the initial message of 2.
decoded = lwe.lwe_decode(decrypted)

assert decoded == 2
```

## Homomorphic Operations

In this section we will show that the LWE scheme above is homomorphic under
addition and multiplication by a plaintext.

More precisely, we will define a homomorphic addition function, $\mathrm{CAdd}$
with the following signature:

\\[ \mathrm{CAdd}: \mathrm{LWE}\_{\mathbf{s}}(m_1) \times
\mathrm{LWE}\_{\mathbf{s}}(m_2) \rightarrow \mathrm{LWE}\_{\mathbf{s}}(m_1 +
m_2) \\]

In other words, $\mathrm{CAdd}$ takes as input encryptions of $m_1$ and $m_2$
and outputs an encryption of $m_1 + m_2$.

Similarly, let $c \in \mathbb{Z}_q$ be an LWE plaintext. We will define a
homomorphic multiplication by $c$ function $\mathrm{CMul}$ function with the
signature:

\\[ \mathrm{CMul}(c, \cdot): \mathrm{LWE}\_{\mathbf{s}}(m) \rightarrow
\mathrm{LWE}\_{\mathbf{s}}(c \cdot m) \\]

In other words, the $\mathrm{CMul}$ function takes as input a plaintext $c$ and
an encryption of $m$ and outputs an encryption of $c\cdot m$.

We will add these operations to our library by implementing the following
methods:

```python
def lwe_add(
    ciphertext_left: LweCiphertext, ciphertext_right: LweCiphertext
) -> LweCiphertext:
    """Homomorphically add two LWE ciphertexts.

    If ciphertext_left is an encryption of m_left and ciphertext_right is
    an encryption of m_right then the output will be an encryption of
    m_left + m_right.
    """
    pass


def lwe_plaintext_multiply(c: int, ciphertext: LweCiphertext) -> LweCiphertext:
    """Homomorphically multiply an LWE ciphertext with a plaintext integer.

    If the ciphertext is an encryption of m then the output will be an
    encryption of c * m.
    """
    pass

```

Before showing how this works let's take a minute to explain why this is both
interesting and useful. Suppose a server hosts a simple linear regression model
with weights whose $w_0$ and $w_1$ linear part is defined by:

\\[ f(x_0, x_1) = w_0\cdot x_0 + w_1 \cdot x_1 \\]

In addition, suppose you have two secret numbers $m_0$ and $m_1$ and you want to
use an untrusted server to compute $f(m_0, m_1)$. Since the server is untrusted,
you do not want to send $m_0$ or $m_1$. Instead you can leverage the homomorphic
addition and multiplication methods above to run the following computation:

```python
# Client side: Generate and encryption key and encrypt m_0 and m_1
key = lwe.generate_lwe_key(config.LWE_CONFIG)
ciphertext_0 = lwe.lwe_encrypt(m_0, key)
ciphertext_1 = lwe.lwe_encrypt(m_1, key)

# Send the ciphertexts to the untrusted server.

# Server side: Generate an encryption of
# f(m_0, m_1) w_0*m_0 + w_1*m_1
ciphertext_result = lwe.lwe_add(
    lwe.lwe_plaintext_multiply(w_0, ciphertext_0),
    lwe.lwe_plaintext_multiply(w_1, ciphertext_1),
)

# The server sends ciphertext_f back to the client

# Client side: Decrypt ciphertext_result to obtain f(m_0, m_1)
result = lwe.lwe_decrypt(ciphertext_result, key)
```

Note that this example can easily be extended to general dot product between a
vector a weights $\mathbf{w}$ and a vector of features $\mathbf{x}$ which is a
core component of neural networks.

### Homomorphic Addition

In this section we'll implement the homomorphic addition function:

\\[ \mathrm{CAdd}: \mathrm{LWE}\_{\mathbf{s}}(m_1) \times
\mathrm{LWE}\_{\mathbf{s}}(m_2) \rightarrow \mathrm{LWE}\_{\mathbf{s}}(m_1 +
m_2) \\]

Let $\mathbf{s}$ be a LWE key, let $m_1, m_2 \in \mathbb{Z}\_q$ be two messages
and let $L_1 = (\mathbf{a}_1, b_1)$ and $L_2 = (\mathbf{a}_2, b_2)$ be
encryptions of $m_1$ and $m_2$ respectively.

By definition, $\mathrm{CAdd}(L_1, L_2)$ should be equal to a ciphertext
$L_{\mathrm{sum}}$ that is an encryption of $m_1 + m_2$.

How can we obtain $L_{\mathrm{sum}}$ from the inputs $L_1$ and $L_2$?

It turns out that all we have to do is add the coefficients of $L_1$ and $L_2$:

\\[ \mathrm{CAdd}(L_1, L_2) := (\mathbf{a}_1 + \mathbf{a}_2, b_1 + b_2) \\]

To prove that $L\_{\mathrm{sum}} = \mathrm{CAdd}(L_1, L_2)$ is indeed an
encryption of $m\_1 + m\_2$, all we have to do is decrypt it:

$$
\begin{align*}
\mathrm{Dec}\_{\mathbf{s}}(L_{\mathrm{sum}}) &= \mathrm{Dec}\_{\mathbf{s}}((\mathbf{a}_1 + \mathbf{a}_2,
b_1 + b_2)) \\
&= (b_1 + b_2) - (\mathbf{a}_1 + \mathbf{a}_2) \cdot \mathbf{s} \\
&= (b_1 - \mathbf{a}_1 \cdot \mathbf{s}) + (b_2 - \mathbf{a}_2 \cdot \mathbf{s}) \\
&= \mathrm{Dec}\_{\mathbf{s}}(L_1) + \mathrm{Dec}\_{\mathbf{s}}(L_2) \\
&= (m_1 + e_1) + (m_2 + e_2)\\
&= (m_1 + m_2) + (e_1 + e_2)
\end{align*}
$$

In summary:

$$
\begin{equation}\label{eq:cadd-correctness}
\mathrm{Dec}\_{\mathbf{s}}(L_{\mathrm{sum}}) = (m_1 + m_2) + (e_1 + e_2)
\end{equation}
$$

This proves that $L_{\mathrm{sum}}$ is an encryption of $m_1 + m_2$ with noise
$e_1 + e_2$.

Later in this post we will also use the homomorphic subtraction function
$\mathrm{CSub}$:

\\[ \mathrm{CSub}: \mathrm{LWE}\_{\mathbf{s}}(m_1) \times
\mathrm{LWE}\_{\mathbf{s}}(m_2) \rightarrow \mathrm{LWE}\_{\mathbf{s}}(m_1 -
m_2) \\]

Here are concrete implementations of $\mathrm{CAdd}$ and $\mathrm{CSub}$:

<div>
<a href="https://github.com/lowdanie/tfhe/blob/main/tfhe/lwe.py">tfhe/lwe.py</a>
</div>

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

Here is an example of homomorphic addition:

```python
key = lwe.generate_lwe_key(config.LWE_CONFIG)

# Encode the values 3 and -1 as LWE plaintexts
plaintext_a = lwe.lwe_encode(3)
plaintext_b = lwe.lwe_encode(-1)

# Encrypt both plaintexts
ciphertext_a = lwe.lwe_encrypt(plaintext_a, key)
ciphertext_b = lwe.lwe_encrypt(plaintext_b, key)

# Homomorphically add the ciphertexts.
ciphertext_sum = lwe.lwe_add(ciphertext_a, ciphertext_b)

# Decrypt the ciphertext_sum and then decode it. The result should
# be equal to 3 + (-1) = 2
decrypted = lwe.lwe_decrypt(ciphertext_sum, key)
decoded = lwe.lwe_decode(decrypted)
assert decoded == 2
```

### Homomorphic Multiplication By Plaintext

In this section we'll implement the homomorphic multiplication by plaintext
function:

\\[ \mathrm{CMul}(c, \cdot): \mathrm{LWE}\_{\mathbf{s}}(m) \rightarrow
\mathrm{LWE}\_{\mathbf{s}}(c \cdot m) \\]

Let $\mathbf{s}$ be a LWE key, let $c, m \in \mathbb{Z}\_q$ be two messages and
let $L = (\mathbf{a}, b)$ be an encryption of $m$.

By definition, $\mathrm{CMul}(c, L)$ should be equal to a ciphertext
$L_{\mathrm{prod}}$ that is an encryption of $c \cdot m$.

Inspired by the previous section, you might guess that we could implement
$\mathrm{CMul}$ by element wise multiplication of $c$ with
$L = (\mathbf{a}, b)$:

\\[ \mathrm{CMul}(c, L) := (c \cdot \mathbf{a}, c \cdot b) \\]

As before, we can verify that this is correct by decrypting
$L_{\mathrm{prod}} = \mathrm{CMul}(c, L)$:

<div>
\begin{align*}
\mathrm{Dec}_{\mathbf{s}}(L_{\mathrm{prod}}) &= \mathrm{Dec}_{\mathbf{s}}((c \cdot \mathbf{a}, c \cdot b)) \\
&= (c \cdot b) - (c \cdot \mathbf{a}) \cdot \mathbf{s} \\ 
&= c \cdot (b - \mathbf{a} \cdot \mathbf{s}) \\
&= c \cdot (m + e) \\
&= (c \cdot m) + (c \cdot e)
\end{align*}
</div>

This shows that the result of decrypting $L_{\mathrm{prod}}$ is the product
$c \cdot m$ together with some noise. We'll analyze the noise more closely in
the next section.

Here is an implementation of $\mathrm{CMul}$:

<div>
<a href="https://github.com/lowdanie/tfhe/blob/main/tfhe/lwe.py">tfhe/lwe.py</a>
</div>

```python
def lwe_plaintext_multiply(c: int, ciphertext: LweCiphertext) -> LweCiphertext:
    """Homomorphically multiply an LWE ciphertext with a plaintext integer."""
    return LweCiphertext(
        ciphertext.config,
        np.multiply(c, ciphertext.a, dtype=np.int32),
        np.multiply(c, ciphertext.b, dtype=np.int32),
    )
```

### Noise Analysis

In this section we'll analyze the effects of homomorphic operations on
ciphertext noise.

Let $m_1$ and $m_2$ be LWE plaintexts. Let $L_i$ be an LWE encryption of $m_i$
with noise $e_i$. By equation \ref{eq:cadd-correctness},
$\mathrm{CAdd}(L_1, L_2)$ is an encryption of $m_1 + m_2$ with noise
$e_1 + e_2$. This means that when we homomorphically add two ciphertexts, the
noise in each of the inputs gets added to the result.

For example, let's see what happens when we take an encryption of $0$ and add it
to itself 10 times. We'll repeat the entire process 1000 times and plot the
ciphertext error distribution:

# Ring LWE

## Introduction

The LWE scheme from the previous section was a good start, but it has a few
major issues:

1. It only supports homomorphic addition.
1. It only supports a limited number of homomorphic operations (due to noise).
1. The result of encrypting a message $m \in \mathbb{Z}_q$ is a ciphertext in
   $\mathbb{Z}_q^n \times \mathbb{Z}_q$. For production level security $n$ is
   usually chosen to be around $1000$. This means that the ciphertext is around
   $1000$ times larger than the plaintext which can be a huge issue if the
   plaintext is large. For example if the plaintext is a 1MB image then after
   encrypting each pixel the result would be 1GB.

As we will see, _Ring LWE_ (RLWE) is a variation of LWE with an additional
multiplication operation. This added structure can be used to solve the
ciphertext size issue, and is also an important ingredient towards the solution
of the first to problems.

## Negacyclic Polynomials

The message space of the RLWE scheme will be the ring of polynomials
$\mathbb{Z}_q[x] / (x^n + 1)$. In this section we will describe the ring
$\mathbb{Z}_q[x] / (x^n + 1)$ and in the next section we'll define the RLWE
scheme.

For starters, $\mathbb{Z}_q[x]$ denotes the ring of polynomials with
coefficients in $\mathbb{Z}_q$. As before, we will assume that $q=2^{32}$ and
identify $\mathbb{Z}_q$ with signed integers. In this case $\mathbb{Z}_q[x]$
denotes the ring of polynomials integer coefficients.

An example of an element in $\mathbb{Z}_q[x]$ would be:

\\[ f(x) = 1 + 2x + 5x^3 \\]

We call $\mathbb{Z}_q[x]$ a
[Ring](<https://en.wikipedia.org/wiki/Ring_(mathematics)>) because we can add
and multiply elements of $\mathbb{Z}_q[x]$. For example, if
$f(x) = 1 + 2x + 5x^3$ and $g(x) = 1 + x$ then $f(x) + g(x) = 2 + 3x + 5x^3$ and
$f(x)\cdot g(x) = 1 + 3x + 2x^2 + 5x^3 + 5x^4$.

The highest power of $x$ in a polynomial is called the _degree_. For example,
the degree of $f(x)$ is $3$ and the degree of $g(x)$ is $1$. Polynomials in
$\mathbb{Z}_q[x]$ can have an arbitrarily high degree.

We now turn to the ring $\mathbb{Z}_q[x] / (x^n + 1)$. The denominator $x^n+1$
plays a similar role to the denominator in the definition
$\mathbb{Z}_q = \mathbb{Z} / q\mathbb{Z}$. Just as $\mathbb{Z}_q$ is defined to
be the integers modulo $q$, $\mathbb{Z}_q[x] / (x^n + 1)$ is the set of
polynomials "modulo" the polynomial $x^n + 1$.

What does this mean? We can represent elements of $\mathbb{Z}_q[x] / (x^n + 1)$
by polynomials whose degree is less than $n$. If we are given a polynomial with
a degree larger than $n$, we can find it's representative modulo $x^n + 1$ by
replacing $x^n$ with $-1$ as many times as necessary. For example, if $n=4$ then
we have the following equivalences modulo $x^4 + 1$:

<div>
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

Note that as the monomial pass $x^4$ they loop back to $1$ but with a negative
sign. For this reason polynomials of this form are sometimes called
_negacyclic_. This process is particularly useful when multiplying polynomials
in $\mathbb{Z}_q[x] / (x^n + 1)$ so as to ensure that the final degree remains
less than $n$. For example, let $f(x) = 1 + x^3$ and $g(x) = x$ be elements of
$\mathbb{Z}_q[x] / (x^4 + 1)$. If we multiply them we get

\\[ f(x) \cdot g(x) = x + x^4 = x - 1 = -1 + x\ (\mathrm{mod}\ x^4 + 1) \\]

As another example that will be relevant later on, if $f(x) = 1 + x + x^2 + x^3$
and $g(x) = x^2$ then

<div>
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

## The RLWE Encryption Scheme

In this section we will define the _RLWE Encryption Scheme_ which is very
similar to the LWE encryption scheme but with messages in
$\mathbb{Z}_q[x] / (x^n + 1)$ rather than $\mathbb{Z}_q$.

A secret key in RLWE is a polynomial in $\mathbb{Z}_q[x] / (x^n + 1)$ with
random _binary_ coefficients. For example, if $n=4$ then a secret key may look
like:

\\[ s(x) = 1 + 0\cdot x + 1\cdot x^2 + 1\cdot x^3 = 1 + x^2 + x^3 \\]

To encrypt a message $m(x) \in \mathbb{Z}_q[x] / (x^n + 1)$ with a secret key
$s(x)$ we first sample a polynomial $a(x)\in \mathbb{Z}_q[x]$ with uniformly
random coefficients in $\mathbb{Z}_q$ and an error polynomial
$e(x)\in \mathbb{Z}_q[x]$ with small error coefficients sampled from
$\mathcal{N}_q$. The encryption of $m(x)$ by $s(x)$ is then given by the pair:

\\[ \mathrm{Enc}_{s(x)}(m(x)) = (a(x), a(x)\cdot s(x) + m(x) + e(x)) \\]

To decrypt a ciphertext $(a(x), b(x))$ we compute:

\\[ \mathrm{Dec}_{s(x)}((a(x), b(x))) = b(x) - a(x)\cdot s(x) = m(x) + e(x) \\]

Similarly to LWE, after decrypting a ciphertext $\mathrm{Enc}_{s(x)}(m(x))$ we
get the original message $m(x)$ together with a small amount of noise. As in the
LWE case, if we are interested in noiseless decryptions we can use a
_restricted_ message space of polynomials whose coefficients are in
$\mathbb{Z}_8$ and extend our encoding and decoding functions to polynomials by
applying our original $\mathrm{Encode}$ and $\mathrm{Decode}$ methods to each
coefficient:

<div>
\begin{align*}
    \mathrm{EncodePolynomial}: \mathbb{Z}_8[x] &\rightarrow \mathbb{Z}_q[x] \\
    a_0 + a_1x + \dots a_{n-1}x^{n-1} &\mapsto
    \mathrm{Encode}(a_0) + \mathrm{Encode}(a_1)x + \dots \mathrm{Encode}(a_{n-1})x^{n-1}
\end{align*}
</div>

and

<div>
\begin{align*}
    \mathrm{DecodePolynomial}: \mathbb{Z}_q[x] &\rightarrow \mathbb{Z}_8[x]\\
    a_0 + a_1x + \dots a_{n-1}x^{n-1} &\mapsto
    \mathrm{Decode}(a_0) + \mathrm{Decode}(a_1)x + \dots \mathrm{Decode}(a_{n-1})x^{n-1}
\end{align*}
</div>

Just like with LWE, if we work with elements in the restricted message space
$r(x) \in \mathbb{Z}_8[x]$ and encode them before encryption and decode after
decryption then we end up with the initial $r(x)$ without noise:

\\[
\mathrm{DecodePolynomial}(\mathrm{Dec}\_{s(x)}(\mathrm{Enc}\_{s(x)}(\mathrm{EncodePolynomial}(r(x)))))
= r(x) \\]

Here is an implementation of the RLWE encryption scheme together with the
encoding and decoding functions:

```python
@dataclasses.dataclass
class RlweConfig:
    degree: int  # The messages are in Z_q[x] / (x^degree + 1)
    noise_std: float  # The noise added during encryption.

@dataclasses.dataclass
class RlweEncryptionKey:
    config: RlweConfig
    key: Polynomial

@dataclasses.dataclass
class RlwePlaintext:
    config: RlweConfig
    message: Polynomial

@dataclasses.dataclass
class RlweCiphertext:
    config: RlweConfig
    a: Polynomial
    b: Polynomial

def encode_rlwe(p: Polynomial, config: RlweConfig) -> RlwePlaintext:
    """Encode a polynomial with coefficients in [-4,4) as an RLWE plaintext."""
    encode_coeff = np.array([encode_lwe(i).message for i in p.coeff])
    return RlwePlaintext(
        config=config, message=Polynomial(N=p.N, coeff=encode_coeff))

def decode_rlwe(plaintext: RlwePlaintext) -> Polynomial:
    """Decode an RLWE plaintext to a polynomial with coefficients in [-4,4)."""
    decode_coeff = np.array(
        [decode_lwe(LwePlaintext(i)) for i in plaintext.message.coeff])
    return Polynomial(N=plaintext.message.N, coeff=decode_coeff)

def generate_rlwe_key(config: LweConfig) -> np.ndarray:
    return RlweEncryptionKey(
        config=config,
        key=Polynomial(
            N=config.degree,
            coeff=np.random.randint(
                low=0, high=2, size=config.degree, dtype=np.int32)))

def rlwe_encrypt(
    plaintext: RlwePlaintext, key: RlweEncryptionKey) -> RlweCiphertext:
    a = Polynomial(
        N=key.config.degree,
        coeff=np.random.randint(
            low=INT32_MIN, high=INT32_MAX+1, size=key.config.degree, dtype=np.int32))
    noise = Polynomial(
        N=key.config.degree,
        coeff=np.int32(
            INT32_MAX * np.random.normal(loc=0.0, scale=key.config.noise_std, size=key.config.degree)))

    b = polynomial_add(polynomial_multiply(a, key.key), plaintext.message)
    b = polynomial_add(b, noise)

    return RlweCiphertext(config=key.config, a=a, b=b)

def rlwe_decrypt(ciphertext: RlweCiphertext, key: RlweEncryptionKey) -> RlwePlaintext:
    message = polynomial_subtract(
        ciphertext.b, polynomial_multiply(ciphertext.a, key.key))
    return RlwePlaintext(config=key.config, message=message)
```

Here is an example:

```python
>>> rlwe_config = RlweConfig(degree=1024, noise_std=2**(-20))
>>> rlwe_key = generate_rlwe_key(rlwe_config)
RlweEncryptionKey(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    key=Polynomial(N=1024, coeff=array([1, 0, 1, ..., 1, 1, 0], dtype=int32)))
>>> # Build a polynomial r(x) = x^2 in the restricted message space.
>>> r = build_monomial(c=1, i=2, N=rlwe_config.degree)
Polynomial(N=1024, coeff=array([0, 0, 1, ..., 0, 0, 0], dtype=int32))
>>> # Encode r(x) as an RLWE plaintext.
>>> rlwe_plaintext = encode_rlwe(r, rlwe_config)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(
        N=1024,
        coeff=array([0, 0, 536870912, ..., 0, 0, 0], dtype=int32)))
>>> # Encrypt the plaintext with the key.
>>> rlwe_ciphertext = rlwe_encrypt(rlwe_plaintext, rlwe_key)
RlweCiphertext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    a=Polynomial(
        N=1024,
        coeff=array([265525746, 339619264, 25096818, ...,  -336116691, 1946582588, -1796375564], dtype=int32)),
    b=Polynomial(
        N=1024,
        coeff=array([2133834000, 1387586285, 1458261587, ..., 405447664, -1104510688, -434404325], dtype=int32)))
>>> # Decrypt the ciphertext. Note that the result is close to the rlwe_plaintext
>>> # with a small amount of noise. I.e, all of the coefficients are small except
>>> # for the third one.
>>> rlwe_decrypted = rlwe_decrypt(rlwe_ciphertext, rlwe_key)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(
        N=1024,
        coeff=array([-3068, -1970, 536872441, ..., 1159, -13, -2289], dtype=int32)))
>>> # Decode rlwe_decrypted to obtain the original r(x) without noise.
>>> decode_rlwe(rlwe_decrypted)
Polynomial(N=1024, coeff=array([0, 0, 1, ..., 0, 0, 0]))
```

## Homomorphic Operations

Similarly to LWE, it is possible to homomorphically add RLWE ciphertexts and
multiply them by plaintext constants.

### Homomorphic Addition

Let $s(x)$ be an RLWE encryption key,
$m_1(x), m_2(x) \in \mathbb{Z}_q[x] / (x^n + 1)$ be RLWE plaintexts and

\\[ \mathrm{Enc}\_{s(x)}(m_i(x)) = (a_i(x), b_i(x)) \in \left( \mathbb{Z}\_q[x]
/ (x^n + 1) \right)^2 \\]

encryptions of $m_i(x)$ with $s(x)$. Just as in the LWE case, we can generate an
encryption of $m_1(x) + m_2(x)$ by adding $\mathrm{Enc}\_{s(x)}(m_1(x))$ and
$\mathrm{Enc}\_{s(x)}(m_2(x))$ element wise:

\\[ \mathrm{Enc}\_{s(x)}(m_1(x) + m_2(x)) = (a_1(x) + a_2(x), b_1(x) + b_2(x))
\\]

DECRYPT

Here is the corresponding code:

```python
def rlwe_add(
    ciphertext_left: RlweCiphertext,
    ciphertext_right: RlweCiphertext) -> RlweCiphertext:
    """Homomorphic RLWE addition.

       If ciphertext_left is an encryption of message_left and ciphertext_right
       is an encryption of message_right, then return an encryption of
       message_left + message_right.
    """
    return RlweCiphertext(
        ciphertext_left.config,
        polynomial_add(ciphertext_left.a, ciphertext_right.a),
        polynomial_add(ciphertext_left.b, ciphertext_right.b))

def rlwe_subtract(
    ciphertext_left: RlweCiphertext,
    ciphertext_right: RlweCiphertext) -> RlweCiphertext:
    """Homomorphic RLWE subtraction.

       If ciphertext_left is an encryption of message_left and ciphertext_right
       is an encryption of message_right, then return an encryption of
       message_left - message_right.
    """
    return RlweCiphertext(
        ciphertext_left.config,
        polynomial_subtract(ciphertext_left.a, ciphertext_right.a),
        polynomial_subtract(ciphertext_left.b, ciphertext_right.b))
```

Here is an example:

```python
>>> rlwe_config = RlweConfig(degree=1024, noise_std=2**(-20))
>>> rlwe_key = generate_rlwe_key(rlwe_config)

>>> # Create two messages
>>> m_1 = build_monomial(c=1, i=1, N=rlwe_config.degree)
Polynomial(N=1024, coeff=array([0, 1, 0, ..., 0, 0, 0], dtype=int32))
>>> m_2 = build_monomial(c=2, i=2, N=rlwe_config.degree)
Polynomial(N=1024, coeff=array([0, 0, 2, ..., 0, 0, 0], dtype=int32))

>>> # Encode them as RLWE plaintexts
>>> rlwe_plaintext_1 = encode_rlwe(m_1, rlwe_config)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(N=1024, coeff=array([0, 536870912, 0, ..., 0, 0, 0], dtype=int32)))
>>> rlwe_plaintext_2 = encode_rlwe(m_2, rlwe_config)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(N=1024, coeff=array([0, 0, 1073741824, ..., 0, 0, 0], dtype=int32)))

>>> # Encrypt the plaintexts
>>> rlwe_ciphertext_1 = rlwe_encrypt(rlwe_plaintext_1, rlwe_key)
RlweCiphertext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    a=Polynomial(N=1024, coeff=array([ -400522947, 780160233, -1240339714, ..., -1201843621, -921665597,   782809419], dtype=int32)),
    b=Polynomial(N=1024, coeff=array([1518284929, 1076194805, 682351634, ..., -2051751750, 1146815193, -1616790507], dtype=int32)))
>>> rlwe_ciphertext_2 = rlwe_encrypt(rlwe_plaintext_2, rlwe_key)
RlweCiphertext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    a=Polynomial(N=1024, coeff=array([-972658319, -948445572, 1516104424, ..., -523471090, -213239951,
       -456683796], dtype=int32)),
    b=Polynomial(N=1024, coeff=array([1362163316, -70198416, -1400655871, ..., -1427181350, 529214435,  1280406048], dtype=int32)))

>>> # Homomorphically add the ciphertexts.
>>> rlwe_ciphertext_sum = rlwe_add(rlwe_ciphertext_1, rlwe_ciphertext_2)
RlweCiphertext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    a=Polynomial(N=1024, coeff=array([-1373181266, -168285339, 275764710, ..., -1725314711, -1134905548,   326125623], dtype=int32)),
    b=Polynomial(N=1024, coeff=array([-1414519051,  1005996389, -718304237, ..., 816034196, 1676029628,  -336384459], dtype=int32)))

>>> # Decrypt the ciphertext of the sum.
>>> rlwe_decrypted_sum = rlwe_decrypt(rlwe_ciphertext_sum, rlwe_key)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(N=1024, coeff=array([1260,  536871447, 1073742677, ..., -2906, -309, 3050], dtype=int32)))

>>> # Decode the plaintext to remove the noise. The result is m_1(x) + m_2(x)
>>> decode_rlwe(rlwe_decrypted_sum)
Polynomial(N=1024, coeff=array([0, 1, 2, ..., 0, 0, 0]))
```

### Homomorphic Multiplication by Plaintext

Just as with LWE, if
$(a(x), b(x)) \in \left(\mathbb{Z}_q[x] / (x^n + 1)\right)^2$ is an RLWE
encryption of $m(x)$ and $c(x) \in \mathbb{Z}_q[x] / (x^n + 1)$ is a plaintext
then

\\[ (c(x) \cdot a(x), c(x) \cdot b(x)) \\]

is an encryption of $c(x) \cdot m(x)$. DECRYPT An important feature of RLWE
multiplication is that we can use the polynomial structure to homomorphically
rotate the coefficients of $m(x)$ by choosing an appropriate plaintext monomial
$c(x)$. For example, consider the message

\\[ m(x) = 0 + x + 2x^2 + 3x^3 \in \mathbb{Z}\_q[x] / (x^4 + 1) \\]

with coefficients $(0, 1, 2, 3)$ and an encryption $\mathrm{Enc}\_{s(x)}(m(x))$.
If we homomorphically multiply $\mathrm{Enc}_{s(x)}(m(x))$ by $c(x) = x$ then we
get an encryption of

\\[ c(x) \cdot m(x) = x \cdot m(x) = -3 + x^2 + 2x^3 \\] whose coefficients
$(-3, 0, 1, 2)$ are a negacyclic rotation of the original coefficients
$(0, 1, 2, 3)$.

Here is an implementation of homomorphic multiplication followed by an example:

```python
def rlwe_plaintext_multiply(
    c: RlwePlaintext,
    ciphertext: RlweCiphertext) -> RlweCiphertext:
    """Homomorphic multiplication by a plaintext.

       If ciphertext is an encryption of m(x) then return an encryption of
       c(x) * m(x)
    """
    return RlweCiphertext(
        ciphertext.config,
        polynomial_multiply(c.message, ciphertext.a),
        polynomial_multiply(c.message, ciphertext.b))
```

```python
>>> rlwe_config = RlweConfig(degree=1024, noise_std=2**(-20))
>>> rlwe_key = generate_rlwe_key(rlwe_config)

>>> # m(x) = x + 2x^2
>>> m = zero_polynomial(N=rlwe_config.degree)
>>> m.coeff[1] = 1
>>> m.coeff[2] = 2
Polynomial(N=1024, coeff=array([0, 1, 2, 0, ..., 0, 0, 0], dtype=int32))

>>> # c(x) = x
>>> c = build_monomial(c=1, i=1, N=rlwe_config.degree)
Polynomial(N=1024, coeff=array([0, 1, 0, ..., 0, 0, 0], dtype=int32))

>>> # Encode m(x) as an RLWE plaintext.
>>> rlwe_plaintext_m = encode_rlwe(m, rlwe_config)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(N=1024, coeff=array([0,  536870912, 1073741824, ..., 0, 0, 0], dtype=int32)))

>>> # Create a plaintext containing c(x). Note that there is no need to encode
>>> # c(x) since it will not be encrypted.
>>> rlwe_plaintext_c = RlwePlaintext(config=rlwe_config, message=c)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(N=1024, coeff=array([0, 1, 0, ..., 0, 0, 0], dtype=int32)))

>>> # Encrypt m(x)
>>> rlwe_ciphertext_m = rlwe_encrypt(rlwe_plaintext_m, rlwe_key)
RlweCiphertext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    a=Polynomial(N=1024, coeff=array([-844666424, 36686307, -526564388, ...,442029872, -654507604, 422241135], dtype=int32)),
    b=Polynomial(N=1024, coeff=array([1558596496, -2028009664, -607932665, ..., 88956176, -416446807,  1405543399], dtype=int32)))

>>> # Homomorphically multiply the ciphertext with c(x)=x.
>>> rlwe_ciphertext_prod = rlwe_plaintext_multiply(rlwe_plaintext_c, rlwe_ciphertext_m)
RlweCiphertext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    a=Polynomial(N=1024, coeff=array([-422241135, -844666424, 36686307, ..., -791176239, 442029872,-654507604], dtype=int32)),
    b=Polynomial(N=1024, coeff=array([-1405543399, 1558596496, -2028009664, ..., -826447358, 88956176,  -416446807], dtype=int32)))

>>> # Decrypt the encrypted product.
>>> rlwe_decrypt_prod = rlwe_decrypt(rlwe_ciphertext_prod, rlwe_key)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=9.5367431640625e-07),
    message=Polynomial(N=1024, coeff=array([-1260, -207, 536871162, 1073742042, ..., -1165, 459, 1412], dtype=int32)))

>>> # Decode to remove the noise. Note that, as expected, we get coefficients of
>>> # c(x) * m(x) = x * (x + 2x^2) = x^2 + 2x^3
>>> decode_rlwe(rlwe_decrypt_prod)
Polynomial(N=1024, coeff=array([0, 0, 1, 2, 0, ..., 0, 0, 0]))
```

## Ciphertext Size

One of the advantages of RLWE over LWE is that the ratio of ciphertext size to
plaintext size is significantly smaller. Recall that with LWE encryption, a
plaintext is a single integer $m \in \mathbb{Z}_q$ and a ciphertext
$(\mathbf{a}, b) \in \mathbb{Z}_q^n \times \mathbb{Z}_q$ contains $n+1$
integers. Therefore, encrypting a message increases the data size by a factor of
$n+1$ which significant since $n$ is typically around 1000.

Let's see how RLWE compares. An RLWE plaintext is a polynomial
$m(x)\in \mathbb{Z}_q[x]/(x^n+1)$ which has $n$ coefficients. A plaintext is a
tuple $(a(x), b(x)) \in \left(\mathbb{Z}_q[x]/(x^n+1)\right)^2$ which has
$n + n = 2n$ coefficients. So encrypting a message only increases the data size
by a factor of 2 - a significant improvements.

The key difference is that with RLWE we can encrypt a polynomial with $n$
coefficients at once, whereas with LWE we encode one number at a time.

# A Homomorphic Multiplexer

In this section we will show how to homomorphically evaluate a
[Multiplexer](https://en.wikipedia.org/wiki/Multiplexer) gate. Specifically, we
will homomorphically evaluate a 2-1 multiplexer:

IMAGE

A standard (plaintext) 2-1 multiplexer $\mathrm{Mux}(b, l_0, l_1)$ has three
inputs: a _selector_ bit $b$ and two _lines_ $l_0$ and $l_1$. The output is
defined by:

<div>
$$
\mathrm{Mux}(b, l_0, l_1) =
\begin{cases} l_0 & \mathrm{if}\ b = 0 \\
              l_1 & \mathrm{if}\ b = 1
\end{cases}
$$
</div>

As a stepping stone on the way to fully homomorphic encryption, we would like to
be able to evaluate a 2-1 multiplexer on _encrypted_ inputs. Following the TFHE
terminology, we will call the encrypted version a _CMux_ gate:

IMAGE

Just like the $\mathrm{Mux}$ function, the $\mathrm{CMux}$ function has three
inputs: An encrypted selector bit $\mathrm{Enc}(b)$ and two RLWE encrypted lines
$\mathrm{Enc}\_{s(x)}(l_0(x))$ and $\mathrm{Enc}\_{s(x)}(l_1(x))$. The output is
an encryption $\mathrm{Enc}_{s(x)}(\mathrm{Mux}(b, l_0(x), l_1(x)))$:

\\[ \mathrm{CMux}(\mathrm{Enc}(b), \mathrm{Enc}\_{s(x)}(l_0(x)),
\mathrm{Enc}\_{s(x)}(l_1(x))) = \mathrm{Enc}\_{s(x)}(\mathrm{Mux}(b, l_0(x),
l_1(x))) \\]

Note that, for reasons that will soon become clear, we have not yet specified
the encryption scheme used to encrypt $b$.

The main point is that the $\mathrm{CMux}$ function allows us to use an
encrypted bit $\mathrm{Enc}(b)$ to select one of the two ciphertexts
$\mathrm{Enc}(l_0)$ and $\mathrm{Enc}(l_1)$, without us ever having access to
the underlying plaintext bit $b$, plaintext lines $l_0$ and $l_1$ or the final
result.

How is this possible? First, note that we can express the standard
$\mathrm{Mux}$ function in terms of addition an multiplication:

\begin{equation}\label{eq:mux} \mathrm{Mux}(b, l_0, l_1) = b \cdot (l_1 - l_0) +
l_0 \end{equation}

Therefore, all we need to do to implement the $\mathrm{CMux}$ function is to
homomorphically evaluate the expression \ref{eq:mux}. In a previous section LINK
we saw how to homomorphically evaluate addition and subtraction of RLWE
ciphertexts. We also saw LINK how to homomorphically multiply an RLWE
_plaintext_ with an RLWE ciphertext. This is not quite enough to homomorphically
evaluate the $\mathrm{Mux}$ expression \ref{eq:mux} since we'd like both the
selection bit $b$ and the lines $l_0$ and $l_1$ to be encrypted.

In the next section we will focus on the problem of homomorphically multiplying
two RLWE ciphertexts. We can then combine that with the homomorphic addition of
ciphertexts to implement $\mathrm{CMux}$.

## Homomorphic Ciphertext Multiplication

Let $m_0(x) \in \mathbb{Z}_q[x]/(x^n+1)$ and
$m_1(x) \in \mathbb{Z}_q[x]/(x^n+1)$ be RLWE plaintexts. Let \\[
\mathrm{Enc}\_{s(x)}(m_i(x)) = (a_i(x), b_i(x)) \in \mathbb{Z}\_q[x]/(x^n+1)
\times \mathbb{Z}\_q[x]/(x^n+1) \\]

be an RLWE encryption of $m_i(x)$. Our goal in this section will be to show how
to homomorphically multiply $\mathrm{Enc}\_{s(x)}(m_0(x))$ and
$\mathrm{Enc}\_{s(x)}(m_1(x))$ to produce an encryption of $m_0(x)\cdot m_1(x)$.
In other words, how can we combine $(a_0(x), b_0(x))$ and $(a_1(x), b_1(x))$ to
produce a valid encryption of $m_0(x)\cdot m_1(x)$?

Following our strategy for homomorphic multiplication by a _plaintext_ LINK, a
natural first guess would be to multiply $(a_0(x), b_0(x))$ and
$(a_1(x), b_1(x))$ element-wise:

\\[ (a_0(x)\cdot a_1(x), b_0(x)\cdot b_1(x)) \\]

Unfortunately, $(a_0(x)\cdot a_1(x), b_0(x)\cdot b_1(x))$ is not a valid RLWE
encryption of $m_0(x)\cdot m_1(x)$.

Going back to the drawing board, note that $(a_0(x), b_0(x))$ is a length $2$
row vector and that such a vector "wants" to be multiplied with a $2 \times 2$
matrix. If we knew the plaintext $m_1(x)$ then we could form the $2\times 2$
diagonal matrix:

<div>
$$
M_1 = \left(\begin{matrix}m_1(x) & 0 \\ 0 & m_1(x)\end{matrix}\right)
$$
</div>

and multiply it with $(a_0(x), b_0(x))$:

<div>
$$
(a_0(x), b_0(x)) \cdot M_1 =
(a_0(x), b_0(x)) \cdot \left(\begin{matrix}m_1(x) & 0 \\ 0 & m_1(x)\end{matrix}\right) =
(a_0(x)\cdot m_1(x), b_0(x)\cdot m_1(x))
$$
</div>

We claim that the expression on the right is a valid RLWE encryption of
$m_0(x)\cdot m_1(x)$ if $m_1(x)$ is small. As usual, to prove this we need to
show that $\mathrm{Dec}_{s(x)}((a_0(x)\cdot m_1(x), b_0(x)\cdot m_1(x)))$ is
equal to $m_0(x) \cdot m_1(x)$ plus a small error. Indeed:

<div>
\begin{align*}
\mathrm{Dec}_{s(x)}((a_0(x)\cdot m_1(x), b_0(x)\cdot m_1(x)))
&= b_0(x)\cdot m_1(x) - (a_0(x)\cdot m_1(x))\cdot s(x) \\
&= (b_0(x) - a_0(x)\cdot s(x))\cdot m_1(x) \\
&\overset{\mathrm{(1)}}{=} (m_0(x) + e(x)) \cdot m_1(x) \\
&= (m_0(x) \cdot m_1(x)) + (e(x)\cdot m_1(x))
\end{align*}
</div>

where $e(x)$ is an error polynomial with small coefficients. Equality
$\mathrm{(1)}$ follow from the fact that $(a_0(x), b_0(x))$ is an RLWE
encryption of $m_0(x)$. If we assume that the coefficients of $m_1(x))$ are
small enough then $(e(x)\cdot m_1(x))$ is still a small error.

There are two big problems with this idea:

1. We do not actually know $m_1(x)$, but only the ciphertext
   $\mathrm{Enc}_{s(x)}(m_1(x))$.
1. The coefficients of $m_1(x)$ may be large.

In the next section we will fix the both problems by introducing the _GSW_
encryption scheme.

As we will see, the GSW scheme solves the first problem by adding an "encryption
of zero" to each row of $M$. Adding these encryptions mask the plaintext
$m_1(x)$ so that it is no longer exposed. At the same time, we will see that
this masked matrix is still a valid encryption of $m_0(x)\cdot m_1(x)$.

GSW's solution to the second problem is a bit tricky. The idea is to represent
the coefficients of $m_1(x)$ in terms of a small basis. For example, if we work
with the binary basis then each coefficient would be represented as a binary
number. Even if a coefficient is large, the elements of its binary
representation are small as they are all equal to $0$ or $1$. The GSW scheme
uses this simple fact in a clever way to limit the error introduced by
multiplication with $m_1(x)$ even when $m_1(x)$ is large.

## The GSW Encryption Scheme

SOMEWHERE PUT A NOTE THAT IT IS RECOMMENDED TO SKIP GSW AND CMUL AND JUST ASSUME
THAT THERE IS A GSW ENCRYPTION SCHEME AND CMul: RLWE(f(x)) \times GSW(g(x)) ->
RLWE(f(x)g(x)) ALSO, EXPLICITLY WRITE THIS AS A GOAL BEFORE THE GSW SECTIONS
BELOW

### Encryptions Of Zero

In the previous section we defined

<div>
$$
M_1 = \left(\begin{matrix}m_1(x) & 0 \\ 0 & m_1(x)\end{matrix}\right)
$$
</div>

and showed that

<div>
$$
(a_0(x), b_0(x)) \cdot M_1
$$
</div>

is a valid RLWE encryption of $m_0(x)\cdot m_1(x)$ when $m_1(x)$ is small.
However, our goal is to construct an encryption of $m_0(x)\cdot m_1(x)$ using
only _encryptions_ of $m_0(x)$ and $m_1(x)$. This rules out the use of $M_1$
which contains the plaintext $m_1(x)$.

As promised in the end of the previous section we'll now solve this problem by
adding an "encryption of zero" to each row of $M$.

An _encryption of zero_ is an RLWE encryption of the zero polynomial. By the
definition of RLWE encryption, an encryption of zero is a pair
$\mathbf{z} = (a(x), b(x)) \in \left(\mathbb{Z}_q[x]/(x^n+1)\right)^2$ such
that:

\\[ b(x) = a(x)s(x) + e(x) \\]

where $a(x)$ is uniformly random and e(x) is a small random error polynomial.

Let $\mathbf{z}_i$ be an encryption of zero for $i=0,1$ and let $Z$ to be the
$2\times 2$ matrix whose $i$-th row is $z_i$.

We will soon show that $(a_0(x), b_0(x)) \cdot (M_1 + Z)$ is still an RLWE
encryption of $m_0(x)\cdot m_1(x)$. The idea of the GSW encryption scheme will
be to encrypt a message $m(x)\in \mathbb{Z}_q[x]/(x^n+1)$ with a ciphertext of
the form

\\[ \mathrm{Enc}^{\mathrm{GSW}}\_{s(x)}(m(x)) = m(x)I_{2\times 2} + Z \\]

The point of this encryption scheme is that we can then homomorphically
_multiply_ an RLWE encryption of $m_0(x)$ and a GSW encryption of $m_1(x)$ by
computing $(a_0(x), b_0(x)) \cdot (M_1 + Z)$ which is a valid ciphertext of
$m_0(x)\cdot m_1(x)$. MAKE THIS MORE CLEAR!!!

<div>
\begin{align*}
\mathrm{Dec}_{s(x)}((a_0(x), b_0(x)) \cdot (M_1 + Z))
&= \mathrm{Dec}_{s(x)}((a_0(x), b_0(x)) \cdot M_1 + (a_0(x), b_0(x)) \cdot Z) \\
&= \mathrm{Dec}_{s(x)}((a_0(x), b_0(x)) \cdot M_1) +
    \mathrm{Dec}_{s(x)}(a_0(x)\cdot \mathbf{z}_0(x) + b_0(x)\cdot \mathbf{z}_1(x)) \\
&\overset{\mathrm{(1)}}{=} (m_0(x)\cdot m_1(x) + e(x) \cdot m_1(x)) +  
                           a_0(x)\cdot \mathrm{Dec}_{s(x)}(\mathbf{z}_0(x)) +
                           b_0(x)\cdot \mathrm{Dec}_{s(x)}(\mathbf{z}_1(x)) \\
&= m_0(x)\cdot m_1(x) + m_1(x)\cdot e(x) + a_0(x)e_0(x) + b_0(x)e_1(x)
\end{align*}
</div>

If the coefficients of $m_1(c)$, $a_0(x)$ and $b_0(x)$ are small enough, then
the last line of the equation above is of the form $m_0(x) \cdot m_1(x)$ plus a
small error term which proves that is an RLWE encryption of
$m_0(x) \cdot m_1(x)$. In the next section we will overcome this limitation on
coefficient sizes.

### The Base $p$ Representation

Let $(a(x), b(x))$ be an RLWE encryption of $f(x)$ with noise $e(x)$ and let
$g(x)I_{2\times2} + Z$ be a GSW encryption of $g(x)$ where the rows have noise
$e_0(x)$ and $e_1(x)$. In the previous section we saw that

\\[ ((a(x), b(x)) \cdot (g(x)I_{2\times2} + Z) \\]

is an RLWE encryption of $f(x)g(x)$ with noise: DEFINE NOISE IN A PREVIOUS
SECTION TO BE DECRYPTION - PLAINTEXT

\\[ g(x)\cdot e(x) + a(x)e_0(x) + b(x)e_1(x) \\]

The first noise term is easy to bound in terms of the input $g(x)$. In fact, as
we will see below, in our implementation of the homomorphic NAND gate $g(x)$
will be equal to one of the constant polynomials $0$ or $1$.

The second two terms $a(x)e_0(x)$ and $b(x)e_1(x)$ are problematic since they
involve the randomly generated ciphertext polynomials $a(x)$ and $b(x)$ whose
coefficients can be arbitrary elements in $\mathbb{Z}_q$. The goal of this
section is to reduce this latter source of noise.

To make things concrete, we'll set $n=2$ and $q=16$. Note that because $n=2$,
all polynomials in $\mathbb{Z}_{16}[x] / (x^2 + 1)$ can be represented with two
coefficients: $c_0 + c_1x$. Furthermore, since $q=16$ the coefficients $c_0$ and
$c_1$ will be in $\mathbb{Z}\_{16}$ and can be represented by 4-bit integers in
the interval $[-8,8)$.

Perhaps the simplest way to decrease the magnitude of the error
$a(x)e_0(x) + b(x)e_1(x)$ is to divide the ciphertext $(a(x), b(x))$ by a
constant $p > 1$ inside of $\mathrm{CMul}$. In order for the output to still be
a ciphertext of $f(x)g(x)$, we can balance this out by redefining the GSW
encryption of $g(x)$ to be $p \cdot g(x)I_{2\times 2} + Z$. Together we get:

<div>
\begin{align*}\label{cmul-divide-by-p}
\mathrm{CMul}((a(x), b(x)), p \cdot g(x)I_{2\times 2} + Z) &:=
\frac{1}{p}(a(x), b(x)) \cdot (p \cdot g(x)I_{2\times 2} + Z) \\
&= (a(x), b(x))\cdot g(x)I_{2\times 2} + \frac{1}{p}(a(x), b(x))\cdot Z \\
&= g(x) \cdot (a(x), b(x)) + \frac{1}{p}(a(x), b(x))\cdot Z
\end{align*}
</div>

Following the same derivation as REF, it is not hard to see that this is still
an RLWE encryption of $f(x)g(x)$ but this time with an error of

\\[ g(x)\cdot e(x) + \frac{1}{p}(a(x)e_0(x) + b(x)e_1(x)) \\]

Since we've set $q=16$, the coefficients of $a(x)$ and $b(x)$ are between $-8$
and $7$. So we can choose $p=2^3$ and decrease the noise by a factor of $8$
which is significant compared to $q$.

Unfortunately, $\frac{1}{8}(a(x), b(x))$ is not well defined in
$\mathbb{Z}\_{16}$ since $8$ is not invertible. The best we can do is divide
modulo $2^3$ but then we'd be throwing out the remainder and loosing a lot of
precision. We can fix this by iteratively dividing the remainder modulo $2^2$,
and then the remainder of that modulo $2$. In other words, we can replace the
coefficients of $a(x)$ and $b(x)$ with their _binary representation_.

More precisely, note that we can express a coefficient $c \in [-8, 8)$ as a sum:

\\[ c = c_0 + c_1 \cdot 2 + c_2 \cdot 2^2 + c_3 \cdot 2^3\\]

where $\vert c_i \vert < 2$. Here are two examples:

<div>
\begin{align*}
  -8 &= 0 + 0\cdot 2 + 0 \cdot 2^2 -1 \cdot 2^3 \\
   6 &= 0 + 1\cdot 2 + 1 \cdot 2^2 + 0 \cdot 2^3
\end{align*}
</div>

We call this representation of $c$ by the coefficients $c_0,\ c_1,\ c_2,\ c_3$
the base-$2$ representation. This is similar to the usual binary representation
except that it is extended to include negative numbers.

More generally, the base-$p$ representation of $\mathbb{Z}_q$ represents
elements of $\mathbb{Z}_q$ as a sum of multiples of powers of $p$. Note that
$m = \lfloor \log_p(q) \rfloor$ is the largest integer satisfying
$p^m \leq q/2$. The base-$p$ representation of an element $c\in\mathbb{Z}_q$ is
a sequence $c_0,\dots,c_m$ satisfying $\vert c_i \vert < p$ and:

\\[ c = c_0 + c_1\cdot p + \dots + c_m \cdot p^m \\]

We'll denote the base-$p$ representation of an element $c\in\mathbb{Z}_q$ by
$\mathrm{Base}_p(c)$:

<div>
\begin{align*}
\mathrm{Base}_p: \mathbb{Z}_q &\rightarrow \mathbb{Z}^{\lfloor \log_p(q) \rfloor} \\
                  c &\mapsto (c_0,\dots,c_{\lfloor \log_p(q/2) \rfloor})
\end{align*}
</div>

Later it will be important to recover an element $c\in\mathbb{Z}_q$ from it's
base-$p$ representation $\mathrm{Base}_p$. For this purpose we define the
inverse of $\mathrm{Base}_p$ which takes a sequence
$(c_0,\dots,c_{\lfloor \log_p(q/2) \rfloor})$ and combines them with powers of
$p$ to produce the original constant $c$:

<div>
\begin{align*}
\mathrm{Base}^{-1}_p: \mathbb{Z}^{\lfloor \log_p(q/2) \rfloor} &\rightarrow \mathbb{Z}_q   \\
                  (c_0,\dots,c_{\lfloor \log_p(q/2) \rfloor}) &\mapsto \sum_{i=0}^{i=\lfloor \log_p(q/2) \rfloor} c_ip^i
\end{align*}
</div>

It is easy to check that for all $c\in\mathbb{Z}_q$:

\\[ \mathrm{Base}^{-1}_p(\mathrm{Base}_p(c)) = c \\]

The key idea is that $\mathrm{Base}_p$ is analogous to division by $p$ and its
inverse $\mathrm{Base}^{-1}_p$ is analogous to multiplication by $p$. Unlike
simple division, $\mathrm{Base}_p$ is well defined in $\mathbb{Z}_q$.

Soon we will upgrade our previous attempt at defining $\mathrm{CMul}$ by using
$\mathrm{Base}_p$ and $\mathrm{Base}^{-1}_p$ in place of the division by $p$ and
the multiplication by $p$ operations in \ref{cmul-divide-by-p}.

Here are implementations of $\mathrm{Base}_p$ (`int_to_base_p`) and
$\mathrm{Base}^{-1}_p$ (`base_p_to_int`). To simplify calculations, in this
implementation $q=2^k$ and $p=2^l$ are given in terms of their base-$2$
logarithms $k$ and $l$.

```python
def get_base_p_rep_length(log_q, log_p):
    """Get number of powers of p in the base-p radix decomposition of
       Z_q = [-q/2, q/2).

       log_q: The logarithm of q base 2.
       log_p: The logarithm of p base 2.
    """
    # Get the highest multiple of log_p that is less than or equal to log_q - 1.
    # Note that this is also the highest power of p that is less than or equal to q/2.
    m = (log_q - 1) // log_p

    # Add one to account for the 0-th power p^0=1.
    return m + 1


def int_to_base_p(x: int, log_q: int, log_p: int) -> list[int]:
    """Get the base-p representation of an integer in Z_q.

       log_q: The base 2 logarithm of q.
       log_p: The base 2 logarithm of p.
    """
    num_powers = get_base_p_rep_length(log_q, log_p)

    remainder = x
    rep = []

    for i in reversed(range(num_powers)):
        p_i = 2**(i * log_p) # p^i
        rep.append(remainder // p_i)
        remainder -= rep[-1] * p_i

    return list(reversed(rep))


def base_p_to_int(base_p_rep: list[int], log_q: int, log_p: int):
    """Convert a base-p representation to an integer in Z_q.

       log_q: The base 2 logarithm of q.
       log_p: The base 2 logarithm of p.
    """

    # c_0 + c_1 * p  + ... + c_m * p^m
    return sum(c * 2**(i * log_p) for i, c in enumerate(base_p_rep))
```

Here are some concrete examples where $q=2^8$ and $p=2^2$:

```
>>> base_p_representation(103, log_q=8, log_p=2)
[3, 1, 2, 1]
>>> base_p_to_int([3, 1, 2, 1], log_q=8, log_p=2)
103
>>> base_p_representation(-60, log_q=8, log_p=2)
[0, 1, 0, -1]
>>> base_p_representation(5, log_q=8, log_p=2)
[1, 1, 0, 0]
```

Note that `int_to_base_p` also accepts arrays of integers. Here is the base-2
representation of all elements of $\mathbb{Z}_8$. The `i-th` column is the
base-2 representation of the `i-th` element:

```python
>>> int_to_base_p(np.arange(-4, 4, dtype=np.int32), log_q=3, log_p=1)
[array([ 0,  1,  0,  1, 0, 1, 0, 1], dtype=int32),
 array([ 0,  0,  1,  1, 0, 0, 1, 1], dtype=int32),
 array([-1, -1, -1, -1, 0, 0, 0, 0], dtype=int32)]
```

We can similarly define the base-$p$ representation of a polynomial $f(x)$ to be
a sequence of $m=\lfloor \log_p(q/2) \rfloor$ polynomials $f_0(x),\dots,f_m(x)$
satisfying:

\\[ f(x) = f_0(x) + f_1(x)\cdot p + \dots + f_m(x) \cdot p^m \\]

such that the absolute values of the coefficients of $f_i(x)$ are all less than
$p$.

<div>
\begin{align*}
\mathrm{Base}_p&: \mathbb{Z}_q[x]/(x^n+1) \rightarrow \left(\mathbb{Z}[x]/(x^n+1)\right)^{\log_p(q/2)+1}
\\
    f(x) &\mapsto (f_0(x),\dots,f_{\log_2(q/p)}(x))
\end{align*}
</div>

For example, when $q=16$ and $p=2$:

\\[ \mathrm{Base}_p(7 + 5x) = ((1 + x), 1, (1 + x), 0) \\]

The base-$p$ representation of a polynomial can be computed from the base-$p$
representation of each coefficient:

```python
def polynomial_to_base_p(f: Polynomial, log_q: int, log_p: int) -> Sequence[Polynomial]:
    return [Polynomial(coeff=v, N=f.N) for v in int_to_base_p(f.coeff, log_q=log_q, log_p=log_p)]
```

For example:

```python
>>> # f(x) = x + 2x + 3x^3
>>> f = Polynomial(N=4, coeff=np.arange(4, dtype=np.int32))
>>> f
Polynomial(N=4, coeff=array([0, 1, 2, 3], dtype=int32))
>>> polynomial_to_base_p(f, log_q=3, log_p=1)
[Polynomial(N=4, coeff=array([0, 1, 0, 1], dtype=int32)),
 Polynomial(N=4, coeff=array([0, 0, 1, 1], dtype=int32)),
 Polynomial(N=4, coeff=array([0, 0, 0, 0], dtype=int32))]
```

Now we have everything we need to define our final versions of $\mathrm{CMul}$
and GSW encryption. As above, let $m=\lfloor \log_p(q/2) \rfloor$ be the largest
power of $p$ appearing in a base-$p$ representation of $\mathbb{Z}_q$. We'll
collect the powers of $p$ into a column vector $\mathbf{v}_p$:

<div>
$$    
\mathbf{v}_p =
\left(\begin{matrix}
    1 \\
    p \\
    \vdots \\
    p^m \\
\end{matrix}\right)
$$
</div>

A nice property of $\mathbf{v}_p$ is that if $f(x)$ is a polynomial and
$\mathrm{Base}_p(f(x)) = (f_0(x),\dots,f_m(x))$ is its base-$p$ representation
then

\\[ \mathrm{Base}_p(f(x)) \cdot \mathbf{v}_p = \sum\_{i=0}^m f_i(x)p^i = f(x)
\\]

Let $\mathbf{0}_m$ denote a length $m$ column vector of zeros.

We will define the GSW encryption of a polynomial $g(x)\in\mathbb{Z}_q/(x^N+1)$
to be:

<div>
$$  
\mathrm{Enc}^{\mathrm{GSW}}(g(x)) = 
    g(x)\left(\begin{matrix}
        \mathbf{v}_p & \mathbf{0}_p \\ \mathbf{0}_p & \mathbf{v}_p
    \end{matrix}\right) + Z
$$
</div>

where $Z$ is a $2m \times 2$ matrix and each row of $Z$ is an RLWE encryption of
$0$.

We'll conclude this section by defining a homomorphic multiplication operation
$\mathrm{CMul}$ whose inputs are an RLWE encryption of $f(x)$ and a GSW
encryption of $g(x)$ and whose output is an RLWE encryption of $f(x)g(x)$:

\\[ \mathrm{CMul}: \mathrm{RLWE}(f(x))\times \mathrm{GSW}(g(x)) \rightarrow
\mathrm{RLWE}(f(x)g(x)) \\]

Let $(a(x), b(x)) \in \mathrm{RLWE}(f(x))$ be an RLWE encryption of $f(x)$ with
noise $e(x)$ and let $M \in \mathrm{GSW}(g(x))$ be a GSW encryption of $g(x)$
where the $i$-th row of $M$ has noise $e_i(x)$. $\mathrm{CMul}$ is defined as:

\\[ \mathrm{CMul}((a(x), b(x)), M) := (\mathrm{Base}\_p(a(x)),
\mathrm{Base}\_p(b(x))) \cdot M \\]

We claim that this gives us an RLWE encryption of $f(x)g(x)$ with small noise.

First let's do a sanity check on the dimensions. Since $\mathrm{Base}\_p(a(x))$
and $\mathrm{Base}\_p(b(x))$ are length $m$ row vectors,
$(\mathrm{Base}\_p(a(x)),
\mathrm{Base}\_p(b(x)))$ is a length $2m$ row vector.
When we multiply with the $2m\times 2$ matrix $M$ we get a length $2$ row vector
which is the correct shape for an RLWE ciphertext.

Now let's verify that $(\mathrm{Base}\_p(a(x)),
\mathrm{Base}\_p(b(x))) \cdot M$
is an RLWE encryption of $f(x)g(x)$ with small noise.

Let $(a_0(x),\dots,a_m(x))$ and $(b_0(x),\dots,b_m(x))$ be the base-$p$
representations of $a(x)$ and $b(x)$.

By the definition of GSW encryption,

<div>
$$  
M = 
    g(x)\left(\begin{matrix}
        \mathbf{v}_p & \mathbf{0}_p \\ \mathbf{0}_p & \mathbf{v}_p
    \end{matrix}\right) + Z
$$
</div>

where the $i$-th row of $Z$, $\mathbf{z}_i$, is an RLWE encryption of zero with
noise $e_i(x)$. Therefore, by linearity of $\mathrm{Dec}^{\mathrm{RLWE}}$:

<div>
\begin{align*}
\mathrm{Dec}^{\mathrm{RLWE}}(\mathrm{CMul}((a(x),b(x)), M)) &= 
\mathrm{Dec}^{\mathrm{RLWE}}((\mathrm{Base}_p(a(x)), \mathrm{Base}_p(b(x))) \cdot M) \\ 
&= g(x)\mathrm{Dec}^{\mathrm{RLWE}}((\mathrm{Base}_p(a(x))\cdot \mathbf{v}_p, \mathrm{Base}_p(b(x))\cdot \mathbf{v}_p)) \\
&\ \ + \sum_{i=0}^m a_i(x) \mathrm{Dec}^{\mathrm{RLWE}}(\mathbf{z}_i)
+ \sum_{i=0}^m b_i(x) \mathrm{Dec}^{\mathrm{RLWE}}(\mathbf{z}_{i+m}) \\
&= g(x)\mathrm{Dec}^{\mathrm{RLWE}}((a(x), b(x))) +
\sum_{i=0}^m a_i(x) e_i(x) + 
\sum_{i=0}^m b_i(x) e_{i+m}(x) \\
&= g(x)(f(x) + e(x)) + \sum_{i=0}^m a_i(x) e_i(x) + \sum_{i=0}^m b_i(x) e_{i+m}(x) \\
&= f(x)g(x) + g(x)e(x) + \sum_{i=0}^m a_i(x) e_i(x) + \sum_{i=0}^m b_i(x) e_{i+m}(x)
\end{align*}
</div>

As expected, the decryption of $\mathrm{CMul}((a(x),b(x)), M)$ is equal to
$f(x)g(x)$ with noise:

\\[ g(x)\cdot e(x) + \sum_{i=0}^m a_i(x)e_i(x) + \sum_{i=0}^m b_i(x)e_{i+m}(x)
\\]

Suppose that the coefficients of $e_i(x)$ are less than a constant $\epsilon$.
Since $|a_i|\leq p$ and $|b_i| < p$, the noise in each of the last two terms in
REF is bounded by:

\\[ (m+1) \cdot n \cdot p \cdot \epsilon \\]

In practice we will use $q=2^{32}$, $p=2^8$. Therefore
$m = \lfloor \log_p(q/2) \rfloor = 4$ and so the noise is bounded by
$5 \cdot 2^8 \cdot n \cdot \epsilon$. In contrast, the coefficients of the error
terms in the naive multiplication REF, $a(x)e(x)$ and $b(x)e(x)$, can be as big
as $q\cdot n \cdot \epsilon = 2^{32} \cdot n \cdot \epsilon$ which is
significantly larger.

Here is an implementation of GSW encryption and $\mathrm{CMul}$.

```python
@dataclasses.dataclass
class GswConfig:
    rlwe_config: RlweConfig
    log_p: int  # Homomorphic multiplication will use the base-2^log_p representation.

@dataclasses.dataclass
class GswPlaintext:
    config: GswConfig
    message: Polynomial

@dataclasses.dataclass
class GswCiphertext:
    config: GswConfig
    rlwe_ciphertexts: list[RlweCiphertext]

@dataclasses.dataclass
class GswEncryptionKey:
    config: GswConfig
    key: Polynomial

def convert_rlwe_key_to_gsw(rlwe_key: RlweEncryptionKey, gsw_config) -> GswEncryptionKey:
    return GswEncryptionKey(
        config=gsw_config,
        key=rlwe_key.key)

def gsw_encrypt(plaintext: GswPlaintext, key: GswEncryptionKey) -> GswCiphertext:
    gsw_config = key.config
    num_powers_of_p = get_base_p_rep_length(log_q=32, log_p=gsw_config.log_p)

    # Create 2 RLWE encryptions of 0 for each element of a base-p representation.
    rlwe_key = convert_gws_key_to_rlwe(key)
    rlwe_plaintext_zero = build_zero_rlwe_plaintext(gsw_config.rlwe_config)
    rlwe_ciphertexts = [
        rlwe_encrypt(rlwe_plaintext_zero, rlwe_key)
        for _ in range(2 * num_powers_of_p)
    ]

    # Add multiples p^i * message to the rlwe ciphertexts
    for i in range(num_powers_of_p):
        p_i = 2**(i * gsw_config.log_p)
        scaled_message = polynomial_constant_multiply(p_i, plaintext.message)

        rlwe_ciphertexts[i].a = polynomial_add(rlwe_ciphertexts[i].a, scaled_message)
        b_idx = i + num_powers_of_p
        rlwe_ciphertexts[b_idx].b = polynomial_add(rlwe_ciphertexts[b_idx].b, scaled_message)

    return GswCiphertext(gsw_config, rlwe_ciphertexts)

def gsw_multiply(gsw_ciphertext: GswCiphertext, rlwe_ciphertext: RlweCiphertext) -> RlweCiphertext:
    gsw_config = gsw_ciphertext.config
    rlwe_config = rlwe_ciphertext.config

    # Concatenate the base-p representations of rlwe_ciphertext.a and rlwe_ciphertext.b
    rlwe_base_p = (polynomial_to_base_p(rlwe_ciphertext.a, log_q=32, log_p=gsw_config.log_p) +
                   polynomial_to_base_p(rlwe_ciphertext.b, log_q=32, log_p=gsw_config.log_p))

    # Multiply the row vector rlwe_base_p with the len(rlwe_base_p)x2 matrix gsw_ciphertext.rlwe_ciphertexts.
    rlwe_ciphertext = RlweCiphertext(
        config=rlwe_config,
        a=zero_polynomial(rlwe_config.degree),
        b=zero_polynomial(rlwe_config.degree))

    for i, p in enumerate(rlwe_base_p):
        rlwe_ciphertext.a = polynomial_add(rlwe_ciphertext.a, polynomial_multiply(p, gsw_ciphertext.rlwe_ciphertexts[i].a))
        rlwe_ciphertext.b = polynomial_add(rlwe_ciphertext.b, polynomial_multiply(p, gsw_ciphertext.rlwe_ciphertexts[i].b))

    return rlwe_ciphertext
```

And here's an example of multiplying a GSW ciphertext with an RLWE ciphertext.

```python
>>> rlwe_config = RlweConfig(degree=1024, noise_std=2**(-24))
>>> gsw_config = GswConfig(rlwe_config=rlwe_config, log_p=8)

>>> rlwe_key = generate_rlwe_key(rlwe_config)
>>> gsw_key = convert_rlwe_key_to_gsw(rlwe_key, gsw_config)

>>> # The GSW plaintext is the constant 2.
>>> gsw_plaintext = GswPlaintext(
>>>    config=gsw_config,
>>>    message=build_monomial(2, 0, gsw_config.rlwe_config.degree))
GswPlaintext(config=GswConfig(rlwe_config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08), log_p=8), message=Polynomial(N=1024, coeff=array([2, 0, 0, ..., 0, 0, 0], dtype=int32)))

>>> # The RLWE plaintext is an encoding of the polynomial f=x
>>> f = build_monomial(c=1, i=1, N=rlwe_config.degree)
Polynomial(N=1024, coeff=array([0, 1, 0, ..., 0, 0, 0], dtype=int32))
>>> rlwe_plaintext = encode_rlwe(f, rlwe_config)
RlwePlaintext(
    config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
    message=Polynomial(
        N=1024,
        coeff=array([0, 536870912, 0, ..., 0, 0, 0], dtype=int32)))

>>> # Encrypt the GSW plaintext with GSW encryption.
>>> gsw_ciphertext = gsw_encrypt(gsw_plaintext, gsw_key)
GswCiphertext(
  config=GswConfig(
    rlwe_config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08), log_p=8),
  rlwe_ciphertexts=[
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([271813620, ..., -223040861], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([-876436729, ..., -350975826], dtype=np.int32)))
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([446327432, ..., 1046228314], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([-1936882923, ..., -935751225], dtype=np.int32)))
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([534040503, ..., -880877497], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([-1997778813, ..., 1054914820], dtype=np.int32)))
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([300855065, ..., -459135528], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([47657021, ..., 623109607], dtype=np.int32)))
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([2116664227, ..., -288715527], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([-758472722, ..., 208553013], dtype=np.int32)))
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([-460910052, ..., 2137117421], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([12211265, ..., -1055758062], dtype=np.int32)))
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([-333035085, ..., -1092598547], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([1248437283, ..., -1155447294], dtype=np.int32)))
    RlweCiphertext(
        config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
        a=Polynomial(N=1024, coeff=array([1059875087, ..., -814647771], dtype=np.int32)),
        b=Polynomial(N=1024, coeff=array([472063671, ..., 408893825], dtype=np.int32)))
  ])

>>> # Encrypt the RLWE plaintext with RLWE encryption
>>> rlwe_ciphertext = rlwe_encrypt(rlwe_plaintext, rlwe_key)
RlweCiphertext(
  config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
  a=Polynomial(N=1024, coeff=array([-858994552, ..., -116332580], dtype=np.int32)),
  b=Polynomial(N=1024, coeff=array([-82429073, ..., -96458989], dtype=np.int32)))

>>> # Homomorphically multiply the ciphertexts.
>>> rlwe_ciphertext_prod = gsw_multiply(gsw_ciphertext, rlwe_ciphertext)
RlweCiphertext(
  config=RlweConfig(degree=1024, noise_std=5.960464477539063e-08),
  a=Polynomial(N=1024, coeff=array([1060371389, ..., 17893357], dtype=np.int32)),
  b=Polynomial(N=1024, coeff=array([560671334, ..., 1806593470], dtype=np.int32)))

>>> # Check that when we decrypt and then decode the product ciphertext we get
>>> # 2 * x = 2x
>>> rlwe_plaintext_prod_decoded = decode_rlwe(rlwe_decrypt(rlwe_ciphertext_prod, rlwe_key))
Polynomial(N=1024, coeff=array([0, 2, 0, ..., 0, 0, 0]))
```

## Implementing The Homomorphic Multiplexer

NOTE: PUT THIS BEFORE THE PREVIOUS SECTIONS AS MOTIVATION

Following REF - "Mux with multiplications", we can use homomorphic addition
$\mathrm{CAdd}$, homomorphic subtraction $\mathrm{CSub}$ and homomorphic
multiplication
$\mathrm{CMul} to implement a _homomorphic multiplexer_ $\mathrm{CMux}$ where
the lines are RLWE ciphertexts and the selector bit is a GSW ciphertext:

IMAGE

\\[ \mathrm{CMux}: \mathrm{GSW}(b) \times \mathrm{RLWE}(f_0(x)) \times
\mathrm{RLWE}(f_1(x)) \rightarrow \mathrm{RLWE}(\mathrm{Mux}(b, f_0(x), f_1(x)))
\\]

To do this, we simply replace the operations in REF with their homomorphic
counterparts:

\\[ \mathrm{CMux}(M, \mathbf{c}_0, \mathbf{c}_1) =
\mathrm{CAdd}(\mathrm{CMul}(M, \mathrm{CSub}(\mathbf{c}_1, \mathbf{c}_0)),
\mathbf{c}_1) \\]

Here is an implementation of $\mathrm{CMux}$:

```python
def cmux(gsw_ciphertext: GswCiphertext, rlwe_ciphertext_0: RlweCiphertext, rlwe_ciphertext_1: RlweCiphertext) -> RlweCiphertext:
    return rlwe_add(
        gsw_multiply(gsw_ciphertext, rlwe_subtract(rlwe_ciphertext_1, rlwe_ciphertext_0)),
        rlwe_ciphertext_0)
```

And here is an example:

```python
>>> rlwe_config = RlweConfig(degree=1024, noise_std=2**(-24))
>>> gsw_config = GswConfig(rlwe_config=rlwe_config, log_p=8)

>>> rlwe_key = generate_rlwe_key(rlwe_config)
>>> gsw_key = convert_rlwe_key_to_gsw(rlwe_key, gsw_config)

>>> # Create a GSW plaintext with a selector bit 0
>>> gsw_plaintext = GswPlaintext(config=gsw_config, message=build_monomial(0, 0, gsw_config.rlwe_config.degree))

>>> # The first CMUX line is an RLWE plaintext encoding the polynomial 1.
>>> rlwe_plaintext_0 = encode_rlwe(build_monomial(1, 0, rlwe_config.degree), rlwe_config)
>>> # Create an RLWE plaintext encoding the polynomial 1.

>>> # The second CMUX line is an RLWE plaintext encoding the polynomial x.
>>> rlwe_plaintext_1 = encode_rlwe(build_monomial(1, 1, rlwe_config.degree), rlwe_config)

>>> # Encrypt the GSW selector bit and the RLWE lines.
>>> gsw_ciphertext = gsw_encrypt(gsw_plaintext, gsw_key)
>>> rlwe_ciphertext_0 = rlwe_encrypt(rlwe_plaintext_0, rlwe_key)
>>> rlwe_ciphertext_1 = rlwe_encrypt(rlwe_plaintext_1, rlwe_key)

>>> # Apply CMUX to the ciphertexts.
>>> rlwe_ciphertext_cmux = cmux(
>>>    gsw_ciphertext, rlwe_ciphertext_0, rlwe_ciphertext_1)

>>> # Verify that decrypting rlwe_ciphertext_cmux yields MUX(0, 1, x) = 1.
>>> rlwe_plaintext_cmux_decoded = decode_rlwe(
>>>    rlwe_decrypt(rlwe_ciphertext_cmux, rlwe_key))
Polynomial(N=1024, coeff=array([1, 0, 0, ..., 0, 0, 0]))
```

# Bootstrapping

In the previous sections we defined the LWE, RLWE and GSW encryption schemes
together with homomorphic addition and multiplication operations.

All three encryption schemes introduce noise during encryption. In this section
we'll analyze this noise more carefully. As motivation, we'll revisit the
example from section REF.

In this example we'll use a fixed LWE encryption key $\mathbf{s}$. Let
$(\mathbf{a}_0, b_0) \in \mathrm{LWE}_{\mathbf{s}}(0)$ be an LWE encryption of
$0$. Let $m = \mathrm{Encode}(1) = 2^{29}$ be the encoding of $1$ and let
$(\mathbf{a}_1, b_1) \in \mathrm{LWE}_{\mathbf{s}}(m)$ be an encryption of $m$.
Here are the distributions of the decryptions
$\mathrm{Dec}_{\mathbf{s}}((\mathbf{a}_0, b_0))$ and
$\mathrm{Dec}_{\mathbf{s}}((\mathbf{a}_1, b_1))$:

IMAGE

Since the noise is so small, we can remove the noise from a decryption by
checking if it is closer to $0$ or to $m$. Let's see what happens if we
homomorphically add the encryption of $0$ to the encryption of $m$ by computing

\\[ \mathrm{CAdd}((\mathbf{a}_0, b_0), (\mathbf{a}_1, b_1)) \\]

by the homomorphic property, the result should be an LWE encryption of
$m + 0 = m$. Let's see what happens when we decrypt
$\mathrm{CAdd}((\mathbf{a}_0, b_0), (\mathbf{a}_0, b_0))$:

IMAGE

As expected, the standard deviation of the noise has now roughly doubled. Now
let's see what happens if we homomorphically add 1000 encryptions of zero to an
encryption of 1:

IMAGE

Now the noise is so large that we can no longer distinguish between $0$ and $m$
which means that our ciphertext is useless. The situation with homomorphic
multiplication is even worse since the noise gets _squared_ during each
operation rather than merely _doubled_.

To realize our goal of performing arbitrary homomorphic computations, we're
going to find away around this limit.

In this section we'll solve this problem by a process called _Bootstrapping_. In
a nutshell, bootstrapping makes it possible to "refresh" a ciphertext by
replacing it with a new ciphertext that has a bounded amount of noise. In the
context of our example, bootstrapping would allow us to replace a ciphertext
whose decryptions look like this:

IMAGE

With one whose decryptions look like this:

IMAGE

Importantly, the bound on the noise in the refreshed ciphertext does not depend
on the noise in the original ciphertext. By periodically refreshing our
ciphertexts, we can keep the noise under control and perform an arbitrary number
of homomorphic operations.

In the TFHE framework, bootstrapping is implemented with lower level homomorphic
operations that are interesting on their own right called _The Homomorphic_
_Multiplexer_ (CMux), _Blind Rotation_ and _Sample Extraction_. In the following
sections we'll introduce each of these primitives and conclude by implementing
the bootstrapping function.

## The Homomorphic Multiplexer

MOVE THIS FROM above

## Blind Rotation

### Definition

Consider the negacyclic [REF] polynomial

\\[ f(x) = 1 + 2x + 3x^2 + 4x^3 \in \mathbb{Z}_q / (x^N+1) \\]

If we multiply $f(x)$ by $x^1$ then, due to the negacyclic property we get

\\[ x^1 \cdot f(x) = -4 + x + 2x^2 + 3x^3 \\]

In other words, multiplying by $x$ rotates the coefficients of $f(x)$ one step
to the right:

\\[ (1, 2, 3, 4) \rightarrow (-4, 1, 2, 3) \\]

The negacyclic property that when a coefficient wraps around it gets multiplied
by $-1$. In general, multiplying by $x^i$ will rotate the coefficients of $f(x)$
by $i$ steps. For example:

\\[ x^3 \cdot f(x) = -2 - 3x - 4x^2 + x^3 \\]

We'll formalize this by defining the $\mathrm{Rotate}(i, f(x))$ that rotated the
coefficients of $f(x)$ $i$ steps to the right:

<div>
\begin{align*}
    \mathrm{Rotate}: \mathbb{Z}_q \times \mathbb{Z}_q / (x^N+1)  &\rightarrow \mathbb{Z}_q / (x^N+1) \\
    (i, f(x)) &\mapsto x^{\frac{2N}{q}i} \cdot f(x)
\end{align*}
</div>

_Blind Rotation_ is the homomorphic version of $\mathrm{Rotate}$. Its inputs are
an LWE _encryption_ of the index $i$ together with the plaintext polynomial
$f(x)$. The output is an RLWE _encryption_ of the polynomial $x^i f(x)$:

\\[ \mathrm{BlindRotate}: \mathrm{LWE}(i) \times \mathbb{Z}_q / (x^N+1)
\rightarrow \mathrm{RLWE}(x^i \cdot f(x)) \\]

The goal of this section will be to implement blind rotation.

### Implementation

Let $\mathbf{s}$ be an encryption key, let
$L = (\mathbf{a}, b) \in \mathrm{LWE}_{\mathbf{s}}(i)$ be an LWE encryption of
$i$ and let $f(x)$ be a polynomial. Our goal is to evaluate
$\mathrm{BlindRotate}((\mathbf{a}, b), f(x))$. By definition, this means that we
must use the inputs $(\mathbf{a}, b)$ and $f(x)$ to produce an RLWE encryption
of $x^i f(x)$.

Recall that the LWE decryption operation is defined by:

\\[ \mathrm{Decrypt}^{\mathrm{LWE}}_{\mathbf{s}}((\mathbf{a}, b)) = b -
\mathbf{s} \cdot \mathbf{a} \\]

Since $(\mathbf{a}, b)$ is an encryption of $i$, the decryption
$b -
\mathbf{s} \cdot \mathbf{a}$ is equal to $i + e$ where $e$ is a small error.
Therefore:

\\[ x^{i + e}f(x) = x^{b - \mathbf{s} \cdot \mathbf{a}} = x^b \cdot
\prod_{i=1}^N x^{s_i a_i} \cdot f(x) \\]

Since each $s_i$ is either $0$ or $1$, $x^{s_i a_i}$ is equal to $1$ or
$x^{a_i}$. Therefore, we can evaluate $x^{i+e}$ by iteratively applying a
multiplexer where $s_i$ is the selector bit. More concretely, we'll start with
$g_0(x) := x^b \cdot f(x)$ and inductively define:

\\[ g_i(x) := \mathrm{Mux}(s_i, g_{i-1}(x), x^{a_i}\cdot g_{i-1}(x)) \\]

for $i = 1,\dots,N$. From the definition of $\mathrm{Mux}$ it follows that:

<div>
$$
g_i(x) =
\begin{cases} g_{i-1}(x) & \mathrm{if}\ s_i = 0 \\
              x^{a_i} \cdot g_{i-1}(x) & \mathrm{if}\ s_i = 1
\end{cases}
$$
</div>

Therefore, it is easy to prove by induction that:

\\[ g_N(x) = \prod_{i=1}^N x^{s_i a_i} \cdot x^b \cdot f(x) \\]

By equation EQ it follows that:

\\[ g_N(x) = x^{i + e}\cdot f(x) \\]

Our strategy for producing an RLWE encryption of $x^i \cdot f(x)$ will be to
iteratively produce RLWE encryptions of $g_i(x)$ for $i=1,\dots,N$ by modifying
our inductive equation EQ.

The first modification will be to replace $s_i$ with a GSW encryption of $s_i$
that we will denote by $t_i$:

\\[ t_i := \mathrm{Enc}_{\mathbf{s}}^{\mathrm{GSW}}(s_i) \\]

The second modification is to replace the multiplexer $\mathrm{Mux}$ with the
homomorphic multiplexer $\mathrm{CMux}$ from section REF. Finally, we'll replace
the product $x^{a_i} \cdot g_{i-1}(x)$ with the homomorphic product
$\mathrm{CMul}$. Putting these together, our new inductive equation is:

\\[ R_i := \mathrm{CMux}(t_i, R_{i-1}(x), \mathrm{CMul}(x^{a_i}, R_{i-1})) \\]

for $i=1,\dots,N$ where the initial value $R_0$ is defined to be an RLWE
encryption of $g_0(x)$:

\\[ R_0 := \mathrm{Enc}^{\mathrm{RLWE}}_{\mathbf{s}}(g_0) \\]

We'll now prove by induction that $R_i$ is an RLWE encryption of of $g_i(x)$.
The $i=0$ is simply the definition of $R_0$.

Suppose that $R_{i-1}$ is an RLWE encryption of $g_{i-1}(x)$. By REF, the
homomorphic product $\mathrm{CMul}(x^{a_i}, R_{i-1})$ is an RLWE encryption of
$x^{a_i}g_{i-1}(x)$. Therefore, since $t_i$ is a GSW encryption of $s_i$, by REF
it follows that:
$\mathrm{CMux}(t_i, R_{i-1}(x), \mathrm{CMul}(x^{a_i}, R_{i-1}))$ is an RLWE
encryption of $\mathrm{Mux}(s_i, g_{i-1}(x), x^{a_i}g_{i-1}(x))$. But this by
definition (EQ) is equal to $g_i(x)$ which is what we had to prove.

In particular, from the $i=N$ case it follows that $R_N$ is an RLWE encryption
of $g_N(x) = x^{i+e} \cdot f(x)$. We can therefore implement
$\mathrm{BlindRotate}$ by using equations EQ and EQ to iteratively compute $R_N$
and setting:

\\[ \mathrm{BlindRotate}(L, f(x)) = R_N \\]

Note that our implementation of $\mathrm{BlindRotate}$ relies on the GSW
encryptions $t_i := \mathrm{Enc}_{\mathbf{s}}^{\mathrm{GSW}}(s_i)$. For reasons
that will become apparent later on, we'll call the list $t_1,\dots,t_N$ the
_Bootstrapping Key_.

Here is a concrete implementation of our $\mathrm{BlindRotate}$ algorithm:

CODE

EXAMPLE

### Noise Analysis

Let $L=(\mathbf{a}, b)$ be an LWE ciphertext and $f(x)$ a polynomial. An
interesting property of $\mathrm{BlindRotate}$ is that the noise in the output
ciphertext $\mathrm{BlindRotate}(L, f(x))$ is fixed and does not depend on the
noise of the input ciphertext $L$. In particular,
$\mathrm{BlindRotate}(L, f(x))$ could have _less_ noise than $L$.

To see why this is try we'll again turn to our inductive equation EQ. The key
observation is that the coefficients of $\mathbf{a}$ only appear in the
homomorphic multiplication

\\[ \mathrm{CMul}(x^{a_i}, R_{i-1}) \\]

but multiplying the ciphertext $R_{i-1}$ by $x^{a_i}$ changes the _order_ of the
coefficients in underlying plaintext but not their absolute values. Therefore
the noise of $R_i$ does not depend on $a_i$. By induction, the noise of
$\mathrm{BlindRotate}(L, f(x)) = R_N$ does not depend on $L=(\mathbf{a}, b)$.

## Sample Extraction

### Definition

Consider the polynomial

\\[ f(x) = f_0 + f_1x + \dots + f_{N-1}x^{N-1} \in \mathbb{Z}_q/(x^N+1) \\]

_Sample Extraction_ is the process of homomorphically extracting an LWE
encryption of one of the _coefficients_ of $f(x)$ from an RLWE encryption of
$f(x)$.

More precisely, for a given index $i$, the function $\mathrm{SampleExtract}$
takes as input an RLWE encryption of $f(x)$ and outputs an LWE encryption of
$f_i$:

\\[ \mathrm{SampleExtract}(i, \cdot): \mathrm{RLWE}\_{s(x)}(f(x)) \rightarrow
\mathrm{LWE}_{\mathbf{s}}(f_i) \\]

As in REF, $\mathbf{s}$ is an LWE encryption key and $s(x)$ is the corresponding
RLWE key.

### Implementation

Let $\mathbf{s} = (s_0,\dots,s_{N-1})$ be an LWE encryption key and let
$s(x)=\sum_{i=0}^{N-1}s_ix^i$ be the corresponding RLWE key. Let $(a(x), b(x))$
be an RLWE encryption of $f(x) = \sum_{i=0}^{N-1}f_ix^i$ with the key $s(x)$.

Let $0 \leq i < N$ be an index. How can we convert the RLWE ciphertext
$(a(x), b(x))$ into an LWE encryption of $f_i$ with the key $\mathbf{s}$?

By the definition of RLWE encryption:

\\[ f(x) = b(x) - a(x)s(x) + e(x) \\]

for some small error $e(x)$.

Let $(a(x)s(x))_i$ denote the $i$-th coefficient of $a(x)s(s)$. Since we are
working with negacyclic polynomials LINK TO SECTION, the $i$-th coefficient
satisfies a variation on the typical polynomial
[convolution](https://en.wikipedia.org/wiki/Convolution):

\\[ (a(x)s(x))\_i = \sum_{j=0}^i a_{i-j}s_j - \sum_{i+1}^{N-1}a_{i-j}s_j \\]

Note that if we define the vector $\mathrm{Conv}(a(x), i)$ by:

\\[ \mathrm{Conv}(a(x), i) := (a_i, a_{i-1}, \dots, a_1, a_0, -a_{N-1},
-a_{N-2}, \dots, -a_{i+1}) \\]

Then we can rewrite equation EQ as a dot product:

\\[ (a(x)s(x))\_i = \mathrm{Conv}(a(x), i) \cdot \mathbf{s} \\]

Plugging this into equation EQ gives us:

\\[ f_i = b_i - \mathrm{Conv}(a(x), i) \cdot \mathbf{s} + e_i \\]

But note that the right hand side of the last equation is exactly the LWE
decryption of $(\mathrm{Conv}(a(x), i), b_i)$ with the encryption key
$\mathbf{s}$! This implies that $(\mathrm{Conv}(a(x), i), b_i)$ is an LWE
encryption of $f_i$ with the key $\mathbf{s}$.

In summary, we can implement $\mathrm{SampleExtract}$ by:

\\[ \mathrm{SampleExtract}(i, (a(x), b(x)) = (\mathrm{Conv}(a(x), i), b_i) \\]

Here is a concrete implementation in python:

CODE + EXAMPLE

### Noise Analysis

As before, let $\mathbf{s} = (s_0,\dots,s_{N-1})$ be an LWE encryption key and
let $s(x)=\sum_{i=0}^{N-1}s_ix^i$ be the corresponding RLWE key. Let
$R = (a(x), b(x))$ be an RLWE encryption of $f(x) = \sum_{i=0}^{N-1}f_ix^i$ with
the key $s(x)$ and noise $e(x)$.

We've seen that for an integer $0 \leq i < N$, $\mathrm{SampleExtract}(i, R)$ is
an LWE encryption of $f_i$. In this section we will show that the LWE ciphertext
$\mathrm{SampleExtract}(i, R)$ has the same amount of noise as the RLWE
ciphertext $R$. In other words, sample extraction does not affect the ciphertext
noise.

The claim follows immediately from equation EQ. Indeed, by the definition of
$\mathrm{SampleExtract}$ and equation EQ we have:

<div>
\begin{align*}
\mathrm{Dec}^{\mathrm{LWE}}_{\mathbf{s}}(\mathrm{SampleExtract}(i, R)) &=
b_i - \mathrm{Conv}(a(x), i) \cdot \mathbf{s} \\
&= f_i - e_i
\end{align*}
</div>

Therefore, the ciphertext $\mathrm{SampleExtract}(i, R)$ has noise $e_i$ which
is one of the coefficients of $e(x)$, the noise of $R$.

## Bootstrapping Definition

PUT THIS TOWARDS THE BEGINNING OF THIS SECTION

As explained in the introduction LINK, the goal of bootstrapping is to convert a
possibly noisy ciphertext into a ciphertext with a bounded amount of noise.
Maintaining an upper bound on the ciphertext noise will allow us to chain
together an arbitrary number of homomorphic operations.

In this section we'll define the bootstrapping function more precisely.

Consider the following step function defined on $\mathbb{Z}_q = [-q/2, q/2)$:

<div>
$$
\mathrm{Step}(x) =
\begin{cases} 0 & \mathrm{if}\ \vert x \vert < q/4 \\
              q/4 & \mathrm{else}
\end{cases}
$$
</div>

IMAGE

The $\mathrm{Bootstrap}$ is a homomorphic version of $\mathrm{Step}$:

\\[ \mathrm{Bootstrap}: \mathrm{LWE}(m) \rightarrow
\mathrm{LWE}(\mathrm{Step}(x)) \\]

In other words, it takes as input an LWE encryption of a message
$m\in\mathbb{Z}_q$. If $\vert m \vert < q/4$ then the output will be an
encryption of $q/4$. Otherwise, the output is an encryption of $q/4$. In
addition to homomorphically evaluating $\mathrm{Step}$, a key property of
$\mathrm{Bootstrap}$ is that the noise of its output will have a bound that is
independent on the noise of the input.

For example, suppose that $q=2^{32}$ and that
$m=\mathrm{Encode}(3) = 3\cdot 2^{29}$. Let $L \in \mathrm{LWE}(m)$ be an LWE
encryption of $m$ with a large amount of noise so that its decryption has the
following distribution:

IMAGE

NOTE SAY SOMETHING ABOVE ABOUT THE MEANING OF THESE PICTURES / L IS A RANDOM
VARIABLE

After bootstrapping $L$, we get a new ciphertext $\mathrm{Bootstrap}(L)$ whose
decryptions have the distribution:

IMAGE

As expected, the mean of the distribution is clustered around $q/4 = 2^{30}$ and
the noise has been reduced.

## Bootstrapping Implementation

We now have all the pieces required to implement the bootstrapping process.
Recall from section LINK that $\mathrm{Bootstrap}$ is a homomorphic version of
the step function $\mathrm{Step}$. Our strategy for implementing
$\mathrm{Bootstrap}$ will be to first implement $\mathrm{Step}$ using the
$\mathrm{Rotate}$ function from section LINK. Then we will leverage
$\mathrm{BlindRotate}$, the homomorphic $\mathrm{Rotate}$ function, to implement
$\mathrm{Bootstrap}.

### Building The Step Function From Polynomial Rotations

Consider the polynomial $f(x) \in \mathbb{Z}_q / (x^N+1)$ whose first $N/2$
coefficients are $-1$ and the last $N/2$ coefficients are $1$:

\\[ f(x) = -1 - x - \dots - x^{N/2 - 1} + x^{N/2} + \dots + x^{N-1} \\]

We will later refer to $f(x)$ as the _Test Polynomial_. What happens if we
rotate $f(x)$ by an integer $-N \leq i < N$? Let's start with $i=1$:

\\[\mathrm{Rotate}(f(x), 1) = x \cdot f(x) = -1 - x - \dots - x^{N/2} +
x^{N/2+1} + \dots + x^{N-1} \\]

All of the coefficients get shifted one place to the right, and the last
coefficient $f_{N-1} = 1$ circles back to the beginning and, due to the
negacyclic property, picks up a minus sign to become $-1$. In summary, the
number of negative coefficients has grown by one from $N/2$ to $N/2+1$ and the
number of positive coefficients has decreased by one.

By iterating this process, we can see that if we rotate $f(x)$ by $i=N/2$ then
_all_ of the coefficients will become $-1$:

\\[\mathrm{Rotate}(f(x), N/2) = -1 - x - \dots - x^{N-1} \\]

What happens if we rotate one more time? Since the last coefficient of
$\mathrm{Rotate}(f(x), N/2)$ is $-1$, when we rotate one more time it circles
back to the beginning and picks up a negative sign, turning it into a 1:

\\[\mathrm{Rotate}(f(x), N/2 + 1) = 1 - x - \dots - x^{N-1} \\]

Now the ordering between the positive and negative coefficient has flipped.
$\mathrm{Rotate}(f(x), N/2 + 1)$ starts with a $1$ and ends with $N-1$ $-1$s. If
we rotate $N/2$ more times for a total of $N$ times, then a similar analysis as
before shows that all of the coefficients the resulting polynomial will be $1$:

\\[\mathrm{Rotate}(f(x), N) = 1 + x + \dots + x^{N-1} \\]

For a polynomial $g(x)$, let $\mathrm{Coeff}(g, i)$ denote the $i$-th
coefficient of $g(x)$. INTRODUCE THIS WITH SAMPLE EXTRACT

The above analysis shows that the zeroth coefficient of
$\mathrm{Rotate}(f(x), i)$ is equal to $-1$ when $0 \leq i \leq N/2$ and equal
to $1$ when $N/2 < i \leq N$. In other words:

<div>
$$
\mathrm{Coeff}(\mathrm{Rotate}(f(x), i), 0) =
\begin{cases} -1 & \mathrm{if}\ 0 \leq i \leq N/2 \\
              1 & \mathrm{if}\ N/2 < i \leq N
\end{cases}
$$
</div>

In fact, it is not hard to generalize this to the case where $i$ can be
negative:

<div>
$$
\mathrm{Coeff}(\mathrm{Rotate}(f(x), i), 0) =
\begin{cases} 1 & \mathrm{if}\ -N \leq i \leq -N/2 \\
              -1 & \mathrm{if}\ -N/2 < i \leq N/2 \\
              1 & \mathrm{if}\ N/2 < i < N
\end{cases}
$$
</div>

Note that this looks very similar to the definition of the step function
$\mathrm{Step}$! UPDATE THE DEF OF STEP

The differences are that the domain of $\mathrm{Step}$ is
$\mathbb{Z}_q = [-q/2, q/2)$ rather that $[-N, N)$ and the outputs of
$\mathrm{Step}$ are $$\{0, q/4\}$$ rather than $$\{-1, 1\}$$. We can fix both
problems by rescaling the inputs and translating the output of
$\mathrm{Coeff}(\mathrm{Rotate}(f(x), i), 0)$. For $i \in \mathbb{Z}_q$ we have:

\\[ \mathrm{Step}(i) = \mathrm{Coeff}(\mathrm{Rotate}(\frac{q}{8} \cdot f(x),
\frac{2N}{q} \cdot i), 0) + \frac{q}{8} \\]

### A Homomorphic Step Function

In section LINK we introduced the bootstrapping function $\mathrm{Bootstrap}$ as
a homomorphic version of the step function $\mathrm{Step}$.

In the previous section we expressed $\mathrm{Step}$ in terms of the polynomial
rotation function $\mathrm{Rotate}$ and the coefficient function
$\mathrm{Coeff}$. In sections LINK and LINK we implemented a homomorphic version
of $\mathrm{Rotate}$ called $\mathrm{BlindRotate}$ and a homomorphic version of
$\mathrm{Coeff} called  $\mathrm{SampleExtract}$. We can plug these homomorphic
functions into equation EQ to define a homomorphic version of $\mathrm{Step}$.

Let $\mathrm{L}$ be an LWE ciphertext. We'll define $\mathrm{Bootstrap}(L)$ to
be:

\\[ \mathrm{Bootstrap}(L) =
\mathrm{SampleExtract}(\mathrm{BlindRotate}(\frac{q}{8} \cdot f(x), \frac{2N}{q}
\cdot L), 0) + \frac{q}{8} \\]

Here is a concrete implementation:

CODE

EXAMPLE

### Bootstrapping Noise

Recall that a key property of $\mathrm{Bootstrap}$ is that if $L$ is an LWE
ciphertext then the noise of $\mathrm{Bootstrap}(L)$ should be bounded
independently of the noise in $L$.

This bound follows immediately from our previous noise analyses of the two
bootstrapping building blocks: $\mathrm{BlindRotate}$ and
$\mathrm{SampleExtract}$. In LINK we saw that the noise of
$\mathrm{BlindRotate}(f(x), L)$ is bounded and independent of the input
ciphertext $L$. And in section LINK we saw that $\mathrm{SampleExtract}(R, 0)$
preserves the noise in the input ciphertext $R$. The bound on the noise of
$\mathrm{Bootstrap}(L)$ now follows from these two results together with
equation EQ.

Let's see how this works in practice using our `bootstrap` implementation in the
previous section.

EXAMPLE

# A Fully Homomorphic NAND Gate

We will now use the bootstrapping function to implement the fully homomorphic
NAND gate promised in the introduction.

## Definition

In this section we'll introduce an encoding of boolean values and precisely
define the homomorphic NAND function.

We'll use the $\mathrm{Encode}$ function from section LINK to represent boolean
values as LWE plaintexts. Specifically, we'll represent $\mathrm{True}$ by
$\mathrm{Encode}(2)$ and represent $\mathrm{False}$ by $\mathrm{Encode}(0)$.

Let $b_0$ and $b_1$ denote two booleans with the above representation. The
homomorphic NAND gate $CNAND$ will be defined as:

\\[ \mathrm{CNAND}: \mathrm{LWE}(b_0) \times \mathrm{LWE}(b_1) \rightarrow
\mathrm{LWE}(\mathrm{NAND}(b_0, b_1)) \\]

In other words, if $L_0$ is an LWE encryption of $b_0$ and $L_1$ is an LWE
encryption of $b_1$ then $\mathrm{CNAND}(L_0, L_1)$ is an LWE encryption of
$\mathrm{NAND}(b_0, b_1)$.

Here is the definition of $\mathrm{CNAND}$ in code:

CODE

Here is an example:

## Implementation

We'll now implement the homomorphic NAND function defined in the previous
section. Our strategy will be to express the standard NAND function in terms of
subtraction and the step function from section LINK. We'll then use the
homomorphic versions of these, $\mathrm{CSub}$ and $\mathrm{Bootstrap}$, to
implement a homomorphic NAND function. Finally, an analysis of the noise will
show that this implementation is in fact _fully_ homomorphic.

Recall that our encoding function $\mathrm{Encode}$ maps values from
$\mathbb{Z}_8$ to $\mathbb{Z}_q$. We'll start by expressing NAND in terms of
operations of $\mathbb{Z}_8$:

IMAGE (circle)

As before, we will identify $\mathrm{True}$ with $2$ and $\mathrm{False}$ with
$0$. Let $b_0, b_1 \in \\{0, 2\\}$ be two boolean values and consider the
expression:

\\[F(b_0, b_1) = -3 - b_0 - b_1 \\]

In terms of the diagram above, we start at $-3$ and take two steps counter
clockwise for each $b_i$ that is "true". Here are the four possible values
(modulo $8$) that this expression can take:

| $b_0$ | $b_1$ | $F(b_0, b_1)$ |
| ----- | ----- | ------------- |
| $0$   | $0$   | $-3$          |
| $0$   | $2$   | $3$           |
| $2$   | $0$   | $3$           |
| $2$   | $2$   | $1$           |

Note that $\vert F(b_0, b_1) \vert < 2$ if and only if $b_0 = b_1 = 2$. We can
therefore upgrade $F(b_0, b_1)$ to a NAND function by composing it with a step
function $S(x)$ defined by:

<div>
$$
S(x) =
\begin{cases} 0 & \vert x \vert < 2 \\
              2 & \mathrm{else}
\end{cases}
$$
</div>

Here's what happens when we apply $S(x)$ to $F(b_0, b_1)$:

| $b_0$ | $b_1$ | $F(b_0, b_1)$ | $S(F(b_0, b_1))$ |
| ----- | ----- | ------------- | ---------------- |
| $0$   | $0$   | $-3$          | $2$              |
| $0$   | $2$   | $3$           | $2$              |
| $2$   | $0$   | $3$           | $2$              |
| $2$   | $2$   | $1$           | $0$              |

As expected, it is evident from the table that

\\[ NAND(b_0, b_1) = S(F(b_0, b_1)) = S(3 - b_0 - b_1) \\]

Returning to the LWE plaintext space $\mathrm{Z}_q$, let $b_0$ and $b_1$ be two
boolean values using the original representation where $\mathrm{True}$ is
represented by $\mathrm{Encode}(2)$ and $\mathrm{False}$ by
$\mathrm{Encode}(0)$. We can similarly express $\mathrm{NAND}(b_0, b_1)$ in
terms of the step function $\mathrm{Step}$ from section LINK:

\\[ \mathrm{NAND}(b_0, b_1) = \mathrm{Step}(\mathrm{Encode}(-3) - b_0 - b_1) \\]

Finally, we can use our homomorphic step function, $\mathrm{Bootstrap}$, to
upgrade this to a _homomorphic_ NAND function. Let $L_0$ and $L_1$ be LWE
encryptions of $\mathrm{Encode}(0)$ or $\mathrm{Encode}(2)$. Our homomorphic
NAND function is defined by:

\\[ \mathrm{CNAND}(L_0, L_1) = \mathrm{Bootstrap}(
\mathrm{CSub}(\mathrm{CSub}(\mathrm{Encode}(-3), L_0), L_1) ) \\]

Here is a concrete implementation:

CODE

In section LINK we saw that the ... bounded noise ... arbitrary ops ... _fully
homomorphic_

OR example
