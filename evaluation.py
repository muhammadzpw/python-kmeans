import numpy as np
from distance import euclidean
from kmeans import KMeans

def elbow_method(X, k_range = range(1,9), init_centroid = "random", distance="euclidean"):
  X = np.array(X)
  distortion_scores = []
  silhouete_scores = []
  for k in k_range:
    model = KMeans(k = k, init_centroid=init_centroid, distance=distance)
    model.train(X)
    distortion_scores.append(model.inertia)
    silhouete_scores.append(silhouete_score(X, model.centroids))
  return distortion_scores, silhouete_scores

def silhouete_coef(x, centroids):
  distances = []
  if len(centroids) == 1 :
    return 0
  for centroid in centroids:
    distances.append(euclidean(x, centroid))
  distances.sort()
  return (distances[1] - distances[0]) / max(distances[0], distances[1])

def silhouete_score(X, centroids):
  X=np.array(X)
  silhouete_coefs = []
  for point in X:
    silhouete_coefs.append(silhouete_coef(point, centroids))
  return sum(silhouete_coefs) / len(silhouete_coefs)
