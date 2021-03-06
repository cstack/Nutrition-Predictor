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
from utilities import *
import json

# Please excuse use of global variable...
cal_only = False

def GuessTheMean(x_train, t_train, x_test, t_test):
  mean = float(sum(t_train))/len(t_train)
  predictions = [mean for i in range(len(x_test))]
  error = computeError(predictions, t_test)
  return (mean, error)


def GuessTheMeanValidate(x_train, t_train, x_test, t_test):
  (mean, error) = GuessTheMean(x_train, t_train, x_test, t_test)
  return {
    "validation error": error,
    "mean": mean
  }

def GuessTheMeanTest(x_train, t_train, x_test, t_test, params):
  (mean, error) = GuessTheMean(x_train, t_train, x_test, t_test)
  params["testing error"] = error
  params["mean"] = mean
  return params

def GuessTheMedian(x_train, t_train, x_test, t_test):
  median = sorted(t_train)[len(t_train)/2]
  predictions = [median for i in range(len(x_test))]
  error = computeError(predictions, t_test)
  return (median, error)


def GuessTheMedianValidate(x_train, t_train, x_test, t_test):
  (median, error) = GuessTheMean(x_train, t_train, x_test, t_test)
  return {
    "validation error": error,
    "median": median
  }

def GuessTheMedianTest(x_train, t_train, x_test, t_test, params):
  (median, error) = GuessTheMedian(x_train, t_train, x_test, t_test)
  params["testing error"] = error
  params["median"] = median
  return params

def KNearestNeighbors(x_train, t_train, x_test, t_test, num_neighbors, weights = 'distance'):
  knn = neighbors.KNeighborsRegressor(num_neighbors, weights)
  t_out = knn.fit(x_train, t_train).predict(x_test)
  error = computeError(t_out, t_test)
  return (t_out,error, knn)

def KNearestNeighborsUniformValidate(x_train, t_train, x_test, t_test):
  min_error = float("inf")
  best_k = 0
  neighbors = [1, 3, 5, 10, 15]
  for k in neighbors:
    (t_out,error, knn) = KNearestNeighbors(x_train, t_train, x_test, t_test, k, 'uniform')
    if (error < min_error):
      min_error = error
      best_k = k

  return {
    "validation error": min_error,
    "k for neighbors": best_k
  }

def KNearestNeighborsDistanceValidate(x_train, t_train, x_test, t_test):
  min_error = float("inf")
  best_k = 0
  neighbors = [1, 3, 5, 10, 15]
  for k in neighbors:
    (t_out,error, knn) = KNearestNeighbors(x_train, t_train, x_test, t_test, k, 'distance')
    if (error < min_error):
      min_error = error
      best_k = k

  return {
    "validation error": min_error,
    "k for neighbors": best_k
  }

def KNearestNeighborsDistanceTest(x_train, t_train, x_test, t_test, params):

  get_num = re.compile('\d+')
  best_k = int(get_num.search(params["k for neighbors"]).group(0))

  (t_out, error, knn) = KNearestNeighbors(x_train, t_train, x_test, t_test, best_k, 'distance')
  save_model(knn, "KNearestNeighborsTest", len(x_train), cal_only)
  params["testing error"] = error
  return params

def KNearestNeighborsUniformTest(x_train, t_train, x_test, t_test, params):

  get_num = re.compile('\d+')
  best_k = int(get_num.search(params["k for neighbors"]).group(0))

  (t_out, error, knn) = KNearestNeighbors(x_train, t_train, x_test, t_test, best_k, 'uniform')
  save_model(knn, "KNearestNeighborsTest", len(x_train), cal_only)
  params["testing error"] = error
  return params

def GradientBoostingRegression(x_train, t_train, x_test, t_test, loss="lad"):
  model = GradientBoostingRegressor(loss=loss)

  model.fit(x_train, t_train)

  t_out = model.fit(x_train, t_train).predict(x_test)
  error = computeError(t_out, t_test)
  return (t_out, error, model)

def GradientBoostingRegressionValidate(x_train, t_train, x_test, t_test):
  min_error = float("inf")
  best_loss_fn = None
  loss_fns = ["huber", "lad"]
  for loss in loss_fns:
    (t_out,error, model) = GradientBoostingRegression(x_train, t_train, x_test, t_test, loss)
    print loss, error
    if (error < min_error):
      min_error = error
      best_loss_fn = loss

  return {
    "validation error": min_error,
    "loss fn": best_loss_fn
  }

def GradientBoostingRegressionTest(x_train, t_train, x_test, t_test, params):
  loss = params["loss fn"]

  (t_out, error, model) = GradientBoostingRegression(x_train, t_train, x_test, t_test, loss)
  save_model(model, "GradientBoostingRegression", len(x_train), cal_only)
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
  save_model(gp, "GaussianProcessRegression", len(x_train), cal_only)
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

  min_error = -1
  best_params = {}
  for c in C_vals:
    for e in e_vals:
      clf = SVR(C=c, epsilon=e, kernel='linear')
      clf.fit(x_train, t_train)
      p = clf.predict(x_test)
      error = computeError(p, t_test)

      if (min_error == -1 or error < min_error):
        min_error = error
        best_params = {
          "best C" : c,
          "best Epsilon" : e,
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

  clf = SVR(C=best_c, epsilon=best_e, kernel='linear')
  clf.fit(x_train, t_train)
  t_out = clf.predict(x_test)
  error = computeError(t_out, t_test)
  params["testing error"] = error
  save_model(clf, "SupportVectorRegression", len(x_train), cal_only)
  return params

def KMeansPerClusterValidating(x_train, t_train, x_test, t_test):
# run LocallyWeighted for each cluster in predict(x_test), where
# (x_train, t_train) is all of the examples in that cluster
  min_error = -1
  best_cluster_k = -1
  best_neighbor_results = []
  best_estimator = None

  clusters = [10, 20, 30, 40, 50, 60, 80, 100]

  # validate the cluster size
  for k in clusters:
    if k > len(x_train):
      continue

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
    neighbor_results = []
    for i in range(0, k):
      sum_k_for_neighbors = 0
      (x_cluster, t_cluster) = examples_in_cluster(x_train, t_train, i, x_train_labels)
      # only compute the KNearest if there are some examples in the new x_test
      if len(x_cluster) < 2:
        neighbor_results.append(1)
        continue

      # kfold for the num neighbors to use for KNN per cluster
      num_folds = min(len(x_cluster), 5)
      kf = cross_validation.KFold(len(x_cluster), n_folds = num_folds, indices=True)

      for train, test in kf:
        x_train_cluster = [x_cluster[r] for r in train]
        t_train_cluster = [t_cluster[r] for r in train]
        x_test_cluster = [x_cluster[r] for r in test]
        t_test_cluster = [t_cluster[r] for r in test]

      #for j in range(len(x_cluster)):
      #  x_train_cluster = [x_cluster[r] for r in range(len(x_cluster)) if r != j]
      #  t_train_cluster = [t_cluster[r] for r in range(len(t_cluster)) if r != j]

      #  x_test_cluster = [x_cluster[j]]
      #  t_test_cluster = [t_cluster[j]]

        results = KNearestNeighborsValidate(x_train_cluster, t_train_cluster, x_test_cluster, t_test_cluster)
        sum_k_for_neighbors += results["k for neighbors"]

      #avg_k_for_neighbors = sum_k_for_neighbors / len(x_cluster)
      avg_k_for_neighbors = sum_k_for_neighbors / num_folds
      neighbor_results.append(avg_k_for_neighbors)

    # figure out the error for this cluster size
    error = KMeansCluster(x_train, t_train, x_test, t_test, estimator, neighbor_results)

    # store the best cluster size
    if (min_error==-1 or error < min_error):
      min_error = error
      best_cluster_k = k
      best_neighbor_results = neighbor_results
      best_estimator = estimator

  results = {
          "validation error" : min_error,
          "k for clusters": best_cluster_k,
          "best estimator" : best_estimator,
          "neighbor results" : best_neighbor_results
        }
  print results
  return results

def KMeansCluster(x_train, t_train, x_test, t_test, estimator, neighbors_per_cluster):

  x_train_labels = estimator.predict(x_train)
  x_test_labels = estimator.predict(x_test)
  avg_error = 0
  for i in range(len(neighbors_per_cluster)):
    # get the data in this cluster
    (x_train_cluster, t_train_cluster) = examples_in_cluster(x_train, t_train, i, x_train_labels)
    (x_test_cluster, t_test_cluster) = examples_in_cluster(x_test, t_test, i, x_train_labels)

    if (len(x_test_cluster) < 1):
      continue

    # run KNearestNeighbor with the neighbor value for this cluster
    (t_out, error, knn) = KNearestNeighbors(x_train_cluster, t_train_cluster, x_test_cluster, t_test_cluster, neighbors_per_cluster[i])
    avg_error += error * len(x_test_cluster)

  return avg_error / len(x_test)


def KMeansClusterTesting(x_train, t_train, x_test, t_test, params):
  print params
  # parse the params dictionary
  get_num = re.compile('\d+')
  #num_clusters = int(get_num.search(params["k for clusters"]).group(0))
  #num_neighbors = int(get_num.search(params["k for neighbors"]).group(0))
  neighbors_per_cluster = params["neighbor results"]
  estimator = params["best estimator"]

  error = KMeansCluster(x_train, t_train, x_test, t_test, estimator, neighbors_per_cluster)

  save_model(estimator, "KMeansPerCluster", len(x_train), cal_only)
  final_results = {
        "testing error" : error,
        }
  final_results.update(params)
  return final_results

def computeError(t_out, t_test):
  diff = [t_out[i] - t_test[i] for i in range(len(t_out))]
  error = sum([abs(diff[i]/t_test[i]) for i in range(len(t_test))]) / len(t_test)
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
  if "best estimator" in results[0]:
    best_error = min([result['validation error'] for result in results])
    for i in range(len(results)):
      if results[i]['validation error'] == best_error:
        return results[i]

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
  cal_only = False
  if len(sys.argv) == 2 and sys.argv[1] == "-c":
    cal_only = True

  results_file = os.path.expanduser(private_consts.SAVE_DIR)+"joshs_results.txt"
  if cal_only:
    results_file = os.path.expanduser(private_consts.SAVE_DIR)+"joshs_results_cal_only.txt"

  try:
    with open(results_file) as f:
        results = json.loads(f.read())
  except:
    print "No results file. Starting from scratch."
    results = {}

  needToSave = False

  num_examples = generate_data_sizes(10000)
  num_examples.append("final")

  algorithms = [KNearestNeighborsUniformValidate]

  testing_algs = [KNearestNeighborsUniformTest]



  for n in num_examples:
    if n == "final":
      print "Loading data"
      (x,t,vocabulary) = load_data(10000, cal_only)
      for validation_fn, testing_fn in zip(algorithms, testing_algs):
        algorithm = validation_fn.__name__
        if "final" not in results[algorithm]:
          print "Final experiment for %s" % algorithm
          params = results[algorithm]["10000 examples"]
          num_train = int(10000*0.8)
          trials = []
          for i in range(5):
            print "Iteration %d" % i
            print "Splitting data"
            (x,t) = shuffle_data(x,t)
            x_train = x[:num_train]
            t_train = t[:num_train]
            x_test = x[num_train:]
            t_test = t[num_train:]
            print "Training"
            result = testing_fn(x_train, t_train, x_test, t_test, params)
            print "Result:", result
            trials.append({
              "testing error": result["testing error"]
              })
          results[algorithm]["final"] = mergeResults(trials)
          print "Saving results to {0}".format(results_file)
          f = open(results_file, "w")
          f.write(json.dumps(results, indent=4, sort_keys=True))
          f.close()
      return

    # load the pickled data and shuffle it around
    (x,t,vocabulary) = load_data(n, cal_only)
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
          print validation_result
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

        if "best estimator" in results[algorithm][experiment_key]:
          del results[algorithm][experiment_key]["best estimator"]
          results[algorithm][experiment_key]["avg neighbor k used"] = sum([i for i in results[algorithm][experiment_key]["neighbor results"]])/len(results[algorithm][experiment_key]["neighbor results"])
          del results[algorithm][experiment_key]["neighbor results"]


        # print the results to a file after run of the algorithm
        print "Saving results to {0}".format(results_file)
        f = open(results_file, "w")
        f.write(json.dumps(results, indent=4, sort_keys=True))
        f.close()

  print "All models learned"
  print "See {0}".format(results_file)

if __name__ == "__main__":
  if len(sys.argv) == 2 and sys.argv[1] == "-c":
   cal_only = True

  learnAllUnlearnedModels()
