# Singular Value Decomposition (SVD)

The most powerful matrix decomposition — from **Netflix recommendations** to **image compression**.

![Singular Value Decomposition](https://mintcdn.com/devweeekends/EctOBf7SNRME6uOG/images/courses/math-for-ml-linear-algebra/svd-concept.svg)

---

# Singular Value Decomposition (SVD)

## The Magic Behind "People Like You Also Bought..."

### An Everyday Mystery

You buy running shoes on Amazon. Suddenly Amazon knows you might want:

* Protein powder
* A fitness tracker
* Compression socks
* A foam roller

How?

You never searched for those things.
You never told Amazon you're a runner.

Amazon discovered a hidden pattern: **"Runner type" people buy these things together.**

Similarly:

* Spotify knows your music taste after 10 songs
* Netflix predicts movie ratings
* YouTube predicts videos you'll watch

All of these systems rely on **Singular Value Decomposition (SVD)**.

SVD finds **hidden patterns in incomplete data**.

---

## Why SVD Matters

| Data             | Hidden Patterns        | Business Value               |
| ---------------- | ---------------------- | ---------------------------- |
| Purchase history | Customer types         | Personalized recommendations |
| Music listening  | Music taste dimensions | Discover Weekly playlist     |
| Movie ratings    | Genre preferences      | Netflix recommendations      |
| Job applications | Candidate profiles     | Better hiring matches        |
| Photos           | Visual features        | Face recognition             |

Companies like **Amazon, Netflix, Spotify, Google** rely heavily on SVD.

**Estimated Time:** 4–5 hours
**Difficulty:** Intermediate → Advanced
**Prerequisites:** Eigenvalues, PCA

---

# Non-Math Example: Restaurant Preferences

## Problem

Five friends rate 6 restaurants.

```python
ratings = {
    #            Pizza Sushi Steak Salad Taco Ramen
    "Alice": [5,4,0,2,0,0],
    "Bob":   [5,0,5,1,4,0],
    "Carol": [1,5,1,5,2,5],
    "Dave":  [0,4,0,5,0,5],
    "Eve":   [4,0,5,0,5,0],
}
```

Humans quickly notice:

Group 1 likes **hearty food**

* pizza
* steak
* tacos

Group 2 likes **lighter food**

* sushi
* salad
* ramen

SVD discovers these **latent factors automatically**.

---

## Apply SVD

```python
import numpy as np

ratings_matrix = np.array([
    [5,4,0,2,0,0],
    [5,0,5,1,4,0],
    [1,5,1,5,2,5],
    [0,4,0,5,0,5],
    [4,0,5,0,5,0]
])

U, S, VT = np.linalg.svd(ratings_matrix, full_matrices=False)

print(S)
```

Example output:

```
[12.2, 9.8, 2.1, 1.2, 0.3]
```

The first **two factors dominate**.

These represent:

* Hearty eater
* Light eater

---

## Predict Missing Ratings

```python
k = 2

predicted = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
```

SVD can predict unknown ratings.

Example:

* Alice would rate **Steak ≈ 4.2**
* Alice would rate **Tacos ≈ 3.8**

This is exactly how **Netflix recommendations work**.

---

# Mathematical Definition

Any matrix can be decomposed into:

[
A = U \Sigma V^T
]

Where

* **U** → row patterns (users)
* **Σ** → singular values (importance)
* **Vᵀ** → column patterns (items)

Example:

```python
A = np.array([
    [4,0,2],
    [0,3,0],
    [2,0,1]
])

U, S, VT = np.linalg.svd(A)

A_reconstructed = U @ np.diag(S) @ VT
```

---

# SVD vs Eigendecomposition

| Property    | Eigen Decomposition | SVD             |
| ----------- | ------------------- | --------------- |
| Matrix type | Square only         | Any matrix      |
| Formula     | A = VΛV⁻¹           | A = UΣVᵀ        |
| Values      | Eigenvalues         | Singular values |
| Reliability | Not always possible | Always works    |

Relationship:

[
\sigma_i = \sqrt{\lambda_i(A^TA)}
]

---

# Low Rank Approximation

SVD allows approximation using **top-k singular values**.

[
A_k = U_k \Sigma_k V_k^T
]

Example:

```python
U, S, VT = np.linalg.svd(A, full_matrices=False)

k = 2

A_approx = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
```

This creates the **best rank-k approximation**.

---

# Example 1 — Netflix Recommendation System

## Rating Matrix

```python
ratings = np.array([
[5,3,0,1,0,0],
[4,0,0,1,0,0],
[1,1,0,5,4,5],
[0,0,0,4,5,4],
[0,1,5,4,0,0],
])
```

Rows = users
Columns = movies

50% ratings are missing.

Goal:

Predict missing values.

---

## Decompose

```python
U, S, VT = np.linalg.svd(ratings, full_matrices=False)
```

Example singular values:

```
[12.48, 9.51, 1.35, 0.89, 0.12]
```

Top two patterns dominate.

---

## Hidden Factors

Example patterns discovered:

* Factor 1 → **Action preference**
* Factor 2 → **Romance preference**

Users become combinations of these factors.

Netflix uses **~50 latent factors**.

---

# Example 2 — House Pattern Discovery

Simulated dataset:

```python
houses = np.random.randn(100,10)
```

Hidden patterns:

* Luxury homes
* Location quality

Apply SVD:

```python
U, S, VT = np.linalg.svd(houses, full_matrices=False)
```

Top two components represent:

1. Luxury / size
2. Location quality

---

# Example 3 — Student Performance

Dataset:

200 students × 8 subjects.

Hidden patterns:

* STEM aptitude
* Humanities aptitude

```python
U, S, VT = np.linalg.svd(students)
```

Use top-k components to predict missing grades.

---

# SVD vs PCA vs Eigenvalues

| Method      | Input         | Output               | Purpose                     |
| ----------- | ------------- | -------------------- | --------------------------- |
| Eigenvalues | Square matrix | Eigenvectors         | Feature importance          |
| PCA         | Dataset       | Principal components | Dimensionality reduction    |
| SVD         | Any matrix    | U Σ Vᵀ               | Recommendation, compression |

---

# Applications of SVD

## 1 Image Compression

```python
U, S, VT = np.linalg.svd(img)

k = 50

compressed = U[:,:k] @ np.diag(S[:k]) @ VT[:k,:]
```

Compression can reach **80% reduction**.

---

## 2 Noise Reduction

```python
U, S, VT = np.linalg.svd(data)

k = 10

denoised = U[:,:k] @ np.diag(S[:k]) @ VT[:k,:]
```

Small singular values often represent noise.

---

## 3 Latent Semantic Analysis (LSA)

Used in **search engines**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Build a document-term matrix.

Apply SVD to discover **topics**.

---

# Key Takeaways

* SVD decomposes matrices:

[
A = U \Sigma V^T
]

* Works for **any matrix**
* Enables **low-rank approximation**
* Finds **latent factors**
* Used in:

  * recommendation systems
  * image compression
  * search engines
  * noise reduction

---

# Interview Questions

### Difference between PCA and SVD?

PCA performs eigendecomposition on covariance matrices.
SVD works directly on the data matrix and is more numerically stable.

---

### How does Netflix use SVD?

User-movie matrix → decomposed into latent user factors and movie factors.

Prediction:

```
rating ≈ user_vector • movie_vector
```

---

### How choose k?

Common methods:

* 95% variance
* singular value decay
* cross validation

---

# Common Pitfalls

* Confusing U and V matrices
* Choosing wrong rank
* Not normalizing features
* Cold-start problem
* High computational cost

---

# Linear Algebra Toolkit for Machine Learning

You now understand:

* Vectors
* Matrices
* Eigenvalues
* PCA
* SVD

These concepts power:

* neural networks
* recommendation systems
* dimensionality reduction
* search engines


