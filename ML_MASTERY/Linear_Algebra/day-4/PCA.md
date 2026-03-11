# Principal Component Analysis (PCA)

Reduce dimensions while keeping the important information — from **10 features to 3**.

![Principal Component Analysis](https://mintcdn.com/devweeekends/EctOBf7SNRME6uOG/images/courses/math-for-ml-linear-algebra/pca-concept.svg)

---

# Principal Component Analysis (PCA)

## The Big Question

Can we predict house prices with fewer features?

You have **10 house features**, but your model is:

* Slow
* Overfitting
* Hard to visualize

Can you reduce to **3 features while keeping 95% of the information?**

**PCA (Principal Component Analysis)** is the answer — dimensionality reduction using **eigenvectors**.

**Estimated Time:** 3–4 hours
**Difficulty:** Intermediate
**Prerequisites:** Eigenvalues
**Main Example:** House feature reduction
**Supporting Examples:** Student profile compression, Movie recommendation optimization

---

# The Core Idea

## From 10 Features to 3

### Problem

Too many features cause:

* Slow training
* Overfitting
* Hard to visualize
* Expensive data collection

### Solution

Find the **most important directions** in data (principal components).

```python
# Before PCA
X = np.array([[3, 2000, 15, 5, 8, 7, 2, 95, 3, 1500]])

# After PCA
X_reduced = pca.transform(X)  # 3 features
```

---

# Mathematics of PCA

## PCA Algorithm

1. Center the data
2. Compute covariance matrix
3. Find eigenvalues and eigenvectors
4. Sort eigenvalues
5. Project data onto top components

---

## Covariance Matrix

For dataset:

[
X \in \mathbb{R}^{n \times d}
]

Covariance matrix:

[
C = \frac{1}{n-1}(X - \bar{X})^T (X - \bar{X})
]

---

## Eigendecomposition

[
C = V \Lambda V^T
]

Where

* **V** = eigenvectors
* **Λ** = eigenvalues

---

## Projection

[
X_{reduced} = (X - \bar{X}) V_k
]

Where:

* (V_k) contains the **top k eigenvectors**

---

# Variance Explained

Each eigenvalue represents **variance captured** by that component.

[
Variance\ Ratio = \frac{\lambda_i}{\sum \lambda}
]

Example eigenvalues:

| Component | Eigenvalue | Variance |
| --------- | ---------- | -------- |
| PC1       | 5.0        | 55.6%    |
| PC2       | 2.0        | 22.2%    |
| PC3       | 1.5        | 16.7%    |
| PC4       | 0.5        | 5.5%     |

Using **PC1 + PC2 = 77.8% variance**.

---

# Example 1: House Price Prediction

## Dataset

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

np.random.seed(42)

n_houses = 1000

houses = np.random.randn(n_houses, 10)

houses[:, 2] = houses[:, 0] * 500 + 1500
houses[:, 1] = houses[:, 0] * 0.5 + 1.5

print(houses.shape)
```

---

## Apply PCA

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
houses_scaled = scaler.fit_transform(houses)

pca = PCA(n_components=3)

houses_pca = pca.fit_transform(houses_scaled)

print(pca.explained_variance_ratio_)
```

Example output:

```
[0.45, 0.28, 0.15]
```

Total variance retained ≈ **88%**

---

## Component Interpretation

```python
components = pca.components_

features = [
'beds','baths','sqft','age','dist',
'school','crime','walk','park','yard'
]

for i,feature in enumerate(features):
    print(feature, components[0,i])
```

Example interpretation

* **PC1 → House size**
* **PC2 → Location quality**
* **PC3 → Age vs modernization**

---

## Visualization

```python
plt.scatter(houses_pca[:,0], houses_pca[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Houses in PCA Space")
plt.show()
```

---

## Prediction Comparison

```python
from sklearn.linear_model import LinearRegression

prices = np.random.randn(n_houses) * 50000 + 300000

model_full = LinearRegression()
model_full.fit(houses_scaled, prices)

model_pca = LinearRegression()
model_pca.fit(houses_pca, prices)
```

Result:

Nearly **same accuracy with fewer features**.

---

# Example 2: Student Performance

Dataset features:

* study hours
* GPA
* attendance
* sleep
* exercise
* social life
* stress
* motivation

```python
students = np.random.randn(500,8)

students_scaled = scaler.fit_transform(students)

pca = PCA(n_components=3)

students_pca = pca.fit_transform(students_scaled)
```

Variance explained example:

```
[0.38, 0.25, 0.18]
```

Interpretation:

| PC  | Meaning             |
| --- | ------------------- |
| PC1 | Academic engagement |
| PC2 | Well-being          |
| PC3 | Social balance      |

---

# Student Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

clusters = kmeans.fit_predict(students_pca)
```

Clusters might represent:

* High academic / low wellbeing
* Balanced students
* Low academic / high social

---

# Example 3: Movie Recommendation

Dataset:

```
1000 movies × 20 features
```

```python
movies = np.random.randn(1000,20)

movies_scaled = scaler.fit_transform(movies)

pca = PCA(n_components=5)

movies_pca = pca.fit_transform(movies_scaled)
```

Variance retained:

```
85%
```

Interpretation:

| Component | Meaning            |
| --------- | ------------------ |
| PC1       | Blockbuster factor |
| PC2       | Emotional factor   |
| PC3       | Comedy factor      |

---

# Fast Similarity Search

```python
from scipy.spatial.distance import cdist

query = movies_pca[0]

distances = cdist([query], movies_pca, metric='cosine')

similar = np.argsort(distances[0])[1:6]
```

Using **5 dimensions instead of 20** improves speed.

---

# PCA Step-by-Step

## Step 1: Standardize Data

```python
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

X_scaled = (X - X_mean) / X_std
```

---

## Step 2: Covariance Matrix

```python
cov_matrix = np.cov(X_scaled.T)
```

---

## Step 3: Eigenvectors

```python
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

Sort by eigenvalues.

---

## Step 4: Projection

```python
k = 2

principal_components = eigenvectors[:, :k]

X_pca = X_scaled @ principal_components
```

---

# Choosing Number of Components

## Method 1 — Variance Threshold

```python
pca = PCA(n_components=0.95)
```

Keeps components explaining **95% variance**.

---

## Method 2 — Scree Plot

```python
plt.plot(pca_full.explained_variance_ratio_)
```

Look for **elbow point**.

---

## Method 3 — Cumulative Variance

```python
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
```

---

# Key Takeaways

* PCA finds **orthogonal directions of maximum variance**
* Always **standardize data**
* Select components explaining **90–95% variance**
* Useful for:

  * Dimensionality reduction
  * Visualization
  * Noise reduction
  * Feature extraction

---

# PCA Interview Questions

### Why standardize data?

Features with larger scales dominate PCA.

---

### How choose number of components?

Methods:

* 95% variance rule
* Scree plot elbow
* Eigenvalue > 1
* Cross validation

---

### PCA Limitations

* Only captures **linear relationships**
* Hard to interpret components
* Sensitive to **outliers**
* Not ideal for categorical data

---

# Common PCA Mistakes

1. Forgetting standardization
2. Using too few components
3. Applying PCA to categorical features
4. Over-interpreting components
5. Ignoring outliers

---

# Next Topic

**Singular Value Decomposition (SVD)**

Used in:

* Recommendation systems
* Matrix factorization
* Latent semantic analysis

