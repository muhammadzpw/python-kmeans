# python-kmeans
Implementation of K-means Clustering Algorithm using Python with Numpy

## Features
### Simmilarity/Distance Measurements:
You can choose one of bellow distance:
- Euclidean distance
- Manhattan distance
- Cosine distance

### Centroid Initializations:
We implement 2 algorithm to initialize the centroid of each cluster:
- **Random initialization** 
  
  Will generate random value on each point in range of [min_value of data - max_value of data]

- **Naive sharding initialization** 
  
  Inspired from from [this article](https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html)

## Quick Start

Clone this repository and create new python file or jupyter notebook file 

```python3
from kmeans import KMeans

# prepare your data in 2D array
X = [
    [ 1,  2],
    [ 3,  4],
    [ 1,  5],
    [ 8,  9],
    [10,  7],
    [ 4,  3],
    [11,  8]
    ]

# define K-Means model
kmeans_model = KMeans(k=3, init_centroid="naive_sharding", distance="euclidean")
kmeans_model.train(X, max_iteration=10, tolerance=0.01)
# [0, 0, 0, 1, 2, 0, 2]
# cluster0 [array([1, 2]), array([3, 4]), array([1, 5]), array([4, 3])]
# cluster1 [array([8, 9])]
# cluster2 [array([10,  7]), array([11,  8])]

# after training, you can use the model to predict some points
X1 = [
    [ 6,  4],
    [ 9,  3],
    [ 5,  5],
    ]

kmeans_model.predict(X1)
# [0, 2, 0]

# Show your cluster's centroids
kmeans_model.centroids
# [array([2.25, 3.5 ]), array([8., 9.]), array([10.5,  7.5])]
```
