"""
Learn a linear regression model
"""

from sklearn import cross_validation, linear_model, neighbors, tree, gaussian_process
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from collections import Counter

import os, pickle, numpy, sys, random, time, math, re


import private_consts
from load_save_data import *
from utilities import pretty_print_predictions, generate_data_sizes
import json

def KNearestNeighbors(x_train, t_train, x_test, t_test, num_neighbors):
  weights = 'uniform'
  knn = neighbors.KNeighborsRegressor(num_neighbors, weights)
  t_out = knn.fit(x_train, t_train).predict(x_test)
  error = computeError(t_out, t_test)
  return (t_out,error)

def KNearestNeighborsValidate(x_train, t_train, x_test, t_test):
  min_error = float("inf")
  best_k = 0
  neighbors = [1, 3, 5, 10, 15]
  for k in neighbors:
    (t_out,error) = KNearestNeighbors(x_train, t_train, x_test, t_test, k)
    if (error < min_error):
      min_error = error
      best_k = k

  return {
    "validation error": min_error,
    "k for neighbors": best_k
  }

def KNearestNeighborsTest(x_train, t_train, x_test, t_test, params):
  
  get_num = re.compile('\d+')
  best_k = int(get_num.search(params["k for neighbors"]).group(0))

  (t_out, error) = KNearestNeighbors(x_train, t_train, x_test, t_test, best_k)
  params["testing error"] = error
  return params
  

def BayesianRidgeRegression(x_train, t_train, x_test, t_test):
  clf = linear_model.BayesianRidge()
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error,
    "average weight": sum(clf.coef_)/len(clf.coef_)
  }

def BayesianRidgeTesting(x_train, t_train, x_test, t_test, params):
  print params
  return params

def GaussianProcessRegression(x_train, t_train, x_test, t_test):
  deDupe(x_train, t_train)
  gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
  gp.fit(x_train, t_train)
  pred, sigma2_pred = gp.predict(x_test, eval_MSE=True)
  error = computeError(pred, t_test)
  return {
    "error": error,
    "sigma2": sum(sigma2_pred.tolist())/len(sigma2_pred.tolist())
  }

def GaussianProcessTesting(x_train, t_train, x_test, t_test, params):
  print params
  return params

def SupportVectorRegression(x_train, t_train, x_test, t_test):
  C_vals = [0.1, 0.5, 1, 1.5, 2]
  e_vals = [0.01, 0.05, 0.1, 0.2, 0.5]
  kernel_vals = ['linear', 'poly', 'rbf', 'sigmoid']
  
  min_error = -1
  best_params = {}
  for c in C_vals:
    for e in e_vals:
      for k in kernel_vals:
        clf = SVR(C=c, epsilon=e, kernel=k)
        clf.fit(x_train, t_train)
        p = clf.predict(x_test)
        error = computeError(p, t_test)

        if (min_error == -1 or error < min_error):
          min_error = error
          best_params = {
            "best C" : c,
            "best Epsilon" : e,
            "kernel" : k
          }

  results = { "validation error": min_error }
  results.update(best_params)
  print results
  return results

def SupportVectorTesting(x_train, t_train, x_test, t_test, params):
  print params
  # parse the params dictionary
  get_float = re.compile('\d+.\d+')
  best_c = float(get_float.search(params["best C"]).group(0))
  best_e = float(get_float.search(params["best Epsilon"]).group(0))
  get_string = re.compile('[a-fA-F]+')
  best_k = params["kernel"]
  
  clf = SVR(C=best_c, epsilon=best_e, kernel=best_k)
  clf.fit(x_train, t_train)
  t_out = clf.predict(x_test)
  error = computeError(t_out, t_test)
  params["testing error"] = error
  return params

def KMeansPerClusterValidating(x_train, t_train, x_test, t_test):
# run LocallyWeighted for each cluster in predict(x_test), where
# (x_train, t_train) is all of the examples in that cluster
  min_error = -1
  best_k = -1
  min_clusters = min(len(x_test), 35)
  max_clusters = min(50, len(x_test))
  clusters = [10, 20, 30, 40, 50, 60, 80, 100]
  best_neighbor = {}
  
  # validate the cluster size
  for k in clusters:
    neighbor_k_array = []
    estimator = KMeans(n_clusters=k, init='k-means++', n_init=10)
    estimator.fit(x_train, t_train)

    # [k by num_features] array of cluster centers
    x_cluster_centers = estimator.cluster_centers_
    # [1 by num_examples] array of cluster assignments
    x_train_labels = estimator.labels_
    x_test_labels = estimator.predict(x_test)

    # run KNearestNeighbors for each of the k clusters,
    # do a weighted average on the error by cluster size
    avg_error = 0
    sum_k_for_neighbors = 0
    neighbor_results = []
    for i in range(0, k):
      train_examples_in_i = numpy.where(x_train_labels==i)[0]
      test_examples_in_i = numpy.where(x_test_labels==i)[0]

      x_cluster = [x_train[x] for x in train_examples_in_i]
      t_cluster = [t_train[x] for x in train_examples_in_i]
      
      for j in range(len(x_cluster)):
        # only compute the KNearest if there are some examples in the new x_test
        if len(x_cluster) < 2:
          continue
        
        x_train_cluster = [x_cluster[r] for r in range(len(x_cluster)) if r != j]
        t_train_cluster = [t_cluster[r] for r in range(len(t_cluster)) if r != j]
      
        x_test_cluster = [x_cluster[j]]
        t_test_cluster = [t_cluster[j]]

        results = KNearestNeighborsValidate(x_train_cluster, t_train_cluster, x_test_cluster, t_test_cluster)
        #avg_error += results["validation error"] * len(t_test_cluster)
        sum_k_for_neighbors += results["k for neighbors"]
  
      avg_k_for_neighbors = sum_k_for_neighbors / len(x_cluster)
      neighbor_results.append(avg_k_for_neighbors)
  
    # store the best cluster size
    avg_error /= len(t_test)
    if (min_error==-1 or avg_error < min_error):
      min_error = avg_error
      best_k = k
      print neighbor_results
      best_neighbor = mergeResults(neighbor_results, False)
        
  results = {
          "validation error" : min_error,
          "k for clusters": best_k
        }
  results.update(best_neighbor)
  print results
  return results

def KMeansCluster(x_train, t_train, x_test, t_test, estimator, neighbors_per_cluster):

  x_train_labels = estimator.labels_
  x_test_labels = estimator.predict(x_test)
  avg_error = 0
  for i in range(len(neighbors_per_cluster)):

    (x_train_cluster, t_train_cluster) = examples_in_cluster(x_train, t_train, i, x_train_labels)
    (x_test_cluster, t_test_cluster) = examples_in_cluster(x_test, t_test, i, x_train_labels)

    if (len(x_test_cluster) < 1):
      continue

    (t_out, error) = KNearestNeighbors(x_train_cluster, t_train_cluster, x_test_cluster, t_test_cluster, neighbors_per_cluster[i])


def KMeansClusterTesting(x_train, t_train, x_test, t_test, params):
  
  # parse the params dictionary
  get_num = re.compile('\d+')
  num_clusters = int(get_num.search(params["k for clusters"]).group(0))
  num_neighbors = int(get_num.search(params["k for neighbors"]).group(0))
  
  # learn the cluster assignments for the training data
  estimator = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10)
  estimator.fit(x_train, t_train)
  
  # [k by num_features] array of cluster centers
  x_cluster = estimator.cluster_centers_
  # PRINT THESE
  
  # [1 by num_examples] array of cluster assignments
  x_train_labels = estimator.labels_
  x_test_labels = estimator.predict(x_test)
  avg_error = 0
  for i in range(0, num_clusters):
    train_examples_in_i = numpy.where(x_train_labels==i)[0]
    test_examples_in_i = numpy.where(x_test_labels==i)[0]
    
    x_train_cluster = [x_train[x] for x in train_examples_in_i]
    t_train_cluster = [t_train[x] for x in train_examples_in_i]
    x_test_cluster = [x_test[x] for x in test_examples_in_i]
    t_test_cluster = [t_test[x] for x in test_examples_in_i]
    
    # only compute the KNearest if there are some examples in the new x_test
    if (len(x_test_cluster) > 0):
      (t_out, error) = KNearestNeighbors(x_train_cluster, t_train_cluster, x_test_cluster, t_test_cluster, num_neighbors)
      avg_error += (error * (len(x_test_cluster)/float(num_clusters)))

  save_model(estimator, "KMeansPerCluster", len(x_train))
  final_results = {
        "testing clusters" : num_clusters,
        "testing neighbors" : num_neighbors,
        "testing error" : avg_error,
        }
  final_results.update(params)
  return final_results

def computeError(t_out, t_test):
  diff = [t_out[i] - t_test[i] for i in range(len(t_out))]
  error = sum([i**2 for i in diff]) / len(t_out)
  return error

def shuffle_data(x,t):
  order = range(len(x))
  random.shuffle(order)
  x_shuffled = [x[i] for i in order]
  t_shuffled = [t[i] for i in order]
  return (x_shuffled, t_shuffled)

def deDupe(x,t):
  """Ensure no x's repeat. Works in place."""
  d = {}
  i = 0
  while i < len(x):
    if str(x[i]) not in d:
      d.setdefault(str(x[i]), i)
      i += 1
    else:
      x.pop(i)
      t.pop(i)

def mergeResults(results, print_range = True):
  merged = {}
  for key in results[0]:
    # if the values for this key are strings, just take the most common string
    if type(results[0][key]) is str:
      val_list = [results[i][key] for i in range(len(results))]
      c = Counter(val_list)
      val_array = c.most_common(1)
      merged[key] = val_array[0][0]
      continue
    
    mean = sum([result[key] for result in results])/len(results)
    stdev = (sum([(result[key] - mean)**2])/len(results))**0.5
    if print_range:
      merged[key] = "{0} +- {1}".format("%.3f" % mean, "%.3f" % stdev)
    else:
      merged[key] = mean
  return merged

def learnAllUnlearnedModels():
  results_file = os.path.expanduser(private_consts.SAVE_DIR)+"justines_results.txt"
  try:
    with open(results_file) as f:
        results = json.loads(f.read())
  except:
    print "No results file. Starting from scratch."
    results = {}

  needToSave = False

  num_examples = generate_data_sizes(30000)

  algorithms = [KNearestNeighborsValidate, KMeansPerClusterValidating, SupportVectorRegression]

  testing_algs = [KNearestNeighborsTest, KMeansClusterTesting, SupportVectorTesting]

  for n in num_examples:
    # load the pickled data and shuffle it around
    (x,t,vocabulary) = load_data(n)
    (x,t) = shuffle_data(x,t)
    
    # split into 80% training, 20% testing
    percent_train = 0.8
    training_size = int(math.floor(percent_train * n))
    
    # split the training set into 5 folds for cross-validation, use the same set for every algorithm
    kf = cross_validation.KFold(training_size, n_folds=5, indices=True)
    
    for validation_fn, testing_fn in zip(algorithms, testing_algs):
      algorithm = validation_fn.__name__
      if algorithm not in results:
        results[algorithm] = {}
      experiment_key = "{0} examples".format(n)
      if experiment_key not in results[algorithm]:
        print "Running {0} on {1} examples...".format(algorithm, n)
        k_results = []
        # run the algorithm over each <train, validate> pair
        for train, test in kf:
          x_train = [x[i] for i in train]
          t_train = [t[i] for i in train]
          x_val = [x[i] for i in test]
          t_val = [t[i] for i in test]
          start = time.time()
          validation_result = validation_fn(x_train, t_train, x_val, t_val)
          finish = time.time()
          validation_result["time"] = finish - start
          k_results.append(validation_result)
        
        # merge all the results from the folds
        results[algorithm][experiment_key] = mergeResults(k_results)
    
        # split up the testing and training sets
        x_train = x[1:training_size]
        t_train = t[1:training_size]
        x_test = x[training_size:n]
        t_test = t[training_size:n]
    
        # run the algorithm on the testing set with the validated params
        results[algorithm][experiment_key] = testing_fn(x_train, t_train, x_test,
          t_test, results[algorithm][experiment_key])

        # print the results to a file after run of the algorithm
        print "Saving results to {0}".format(results_file)
        f = open(results_file, "w")
        f.write(json.dumps(results, indent=4, sort_keys=True))
        f.close()

  print "All models learned"
  print "See {0}".format(results_file)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    num_examples = 100
  else:
    num_examples = int(sys.argv[1])

  learnAllUnlearnedModels()
