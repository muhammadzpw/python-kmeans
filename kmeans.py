import numpy as np
import math
from distance import euclidean, manhattan, cosine
from time import time

VALID_DISTANCE_ARG = {
  "euclidean": euclidean,
  "manhattan": manhattan,
  "cosine": cosine
}

VALID_INIT_CENTROID_ARG = ["random", "naive_sharding"]

class KMeans():
  '''
    Initialization of KMeans model
    params:
    - k : number of cluster
    - init_centroid : strategy to initialize the centroid. valid arguments: "random", "naive_sharding"
    - distannce : metrics to calculate distance of each point of datum. valid arguments: "euclidean", "manhattan", "cosine"
    '''
  def __init__(self, k=3, init_centroid="random", distance="euclidean"):
    self.k = k
    if init_centroid in VALID_INIT_CENTROID_ARG:
      self.init_centroid = init_centroid
    else:
      raise Exception("init_centroid is not valid")

    if distance in VALID_DISTANCE_ARG.keys():
      self.distance = VALID_DISTANCE_ARG[distance]
    else:
      raise Exception("distance is not valid")

  def choose_random_point(self, X):
    '''
    Pick random point in range of (min_value of X - max_value of X)
    '''
    min_val = np.min(X)
    max_val = np.max(X)
    return np.random.uniform(low=min_val,high=max_val, size=(self.n_features,)) 

  def random_init(self, X):
    '''
    Initialize each cluster's centroid with random point
    '''
    initial_centroids = []
    for _ in range(self.k):
      rand_centroid = self.choose_random_point(X)
      initial_centroids.append(rand_centroid)
    return initial_centroids

  def naive_sharding_init(self, X):
    '''
    Intuition from this article https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
    1. sum each instance and create new column for it
    2. sort by sum column from 1
    3. split into k-equal size, we call it shard
    4. get mean of each shard and make them the centroids of each cluster
    '''
    initial_centroids = []
    # 1
    list_of_instance_sum_tupple = [] 
    for instance in X:
      list_of_instance_sum_tupple.append((np.sum(instance), instance))
    
    # 2 
    list_of_instance_sum_tupple.sort(key=lambda tup: tup[0], reverse=False)

    # 3 & 4
    segment = math.ceil(len(list_of_instance_sum_tupple) / self.k)
    for i in range(self.k):
      # 3
      shard = list_of_instance_sum_tupple[(i * segment):((i+1) * segment)]
      shard = [x[1] for x in shard]
      # 4 mean of shard
      mean_shard = np.zeros(self.n_features)
      for x in shard:
        mean_shard = mean_shard + x
      mean_shard = mean_shard / len(shard)
      initial_centroids.append(mean_shard)

    return initial_centroids

  def train(self, X, max_iteration = 100, tolerance = 0.001, verbose=False):
    '''
    Process to train data into K cluster using KMeans

    params:
    - X : data train (2D array)
    - max_iterations : force condition to stop the training
    - tolerance : stop iteration when the centroid do not change that much 

    '''
    start_time = time()

    X = np.array(X)
    # Validate: matrix X must be 2D array
    if len(X.shape) != 2:
      raise Exception("Data must be 2D array")

    # save the dimension of features 
    self.n_features = X[0].shape[0]

    # Create k cluster and initialize centroid foreach cluster
    self.centroids = []
    if self.init_centroid == "random":
      self.centroids = self.random_init(X)
    else:
      self.centroids = self.naive_sharding_init(X)
  
    if verbose:
      print("initial centroid", self.centroids)

    # Init empty cluster member
    self.cluster_members = [[] for _ in range(self.k)]
    
    # Enter the iteration
    iteration = 0
    total_diff = float("inf")
    while iteration < max_iteration:

      if verbose:
        print("iteration", iteration)
        print("centroid", self.centroids)

      current_cluster_members = [[] for _ in range(self.k)]      
      for data_point in X:
        # print()
        # print(data_point)
        # calculate distance to each centroids
        min_distance = float("inf")
        cluster = None
        for cluster_idx, centroid_i in enumerate(self.centroids):
          distance = self.distance(centroid_i, data_point)
          # print("centroid, distance", centroid_i, distance)
          if distance <= min_distance:
            cluster = cluster_idx
            min_distance = distance
        # the nearest distance will place the point to corresponding cluster
        current_cluster_members[cluster].append(data_point)

      if verbose:
        print("cluster member")
        for idx, ccm in enumerate(current_cluster_members):
          print("cluster" + str(idx), ccm)

      new_centroids = [[] for _ in range(self.k)]
      for cluster_i in range(self.k):
        # Adjust new centroids 
        new_centroid_i = np.zeros(self.n_features)
        members_of_current_cluster = current_cluster_members[cluster_i]
        if len(members_of_current_cluster) > 0:
          for member in current_cluster_members[cluster_i]:
            new_centroid_i = new_centroid_i + member
          new_centroid_i = new_centroid_i / len(members_of_current_cluster) # Get average point from all members
        else:
          # If cluster has no member then pick random point
          new_centroid_i = self.choose_random_point(X)

        new_centroids[cluster_i] = new_centroid_i

      if verbose:
        print("new centroid", new_centroids)

      # Stop Iteration if centroids do not change
      total_diff = float(0.0)
      for cluster_i in range(self.k):
        total_diff = total_diff + self.distance(self.centroids[cluster_i], new_centroids[cluster_i])
      
      self.centroids = new_centroids
      self.cluster_members = current_cluster_members
      
      if verbose:
        print("total diffs:", total_diff)
        print()

      if total_diff <= tolerance:
        break
      iteration = iteration + 1

    if verbose:
      print(self.cluster_members)
      for idx, cm in enumerate(self.cluster_members):
        print("cluster"+ str(idx), cm)
    print("Training time", (time() - start_time) * 100 , "ms")
    print("Stopped at iteration", iteration)
    return self.predict(X)


  def predict(self, X):
    result = []
    for data_point in X:
      # calculate distance to each centroids
      min_distance = float("inf")
      cluster = None
      for cluster_idx, centroid_i in enumerate(self.centroids):
        distance = self.distance(centroid_i, data_point)
        if distance <= min_distance:
          cluster = cluster_idx
          min_distance = distance
      result.append(cluster)
    return result

  # def test(self, X):
  #   X = np.array(X)
  #   # Validate: matrix X must be 2D array
  #   if len(X.shape) != 2:
  #     raise Exception("Data must be 2D array")

  #   for data_point in X:
  #     # calculate distance to each centroids
  #     min_distance = float("inf")
  #     cluster = None
  #     current_cluster_members = [[] for _ in range(self.k)]    
  #     for cluster_idx, centroid_i in enumerate(self.centroids):
  #       distance = self.distance(centroid_i, data_point)
  #       if distance <= min_distance:
  #         cluster = cluster_idx
  #         min_distance = distance
  #     # the nearest distance will place the point to corresponding cluster
  #     current_cluster_members[cluster].append(data_point)
