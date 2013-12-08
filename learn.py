"""
Learn a linear regression model
"""

from sklearn import cross_validation, linear_model, neighbors, tree, gaussian_process
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.cluster import KMeans

import os, pickle, numpy, sys, random, time
import private_consts
from load_save_data import load_data, load_and_split_data
from utilities import pretty_print_predictions
import json

def KNearestWithPCA(x_train, t_train, x_test, t_test, num_components=400):
  pca = PCA(n_components=num_components)
  new_x_train = pca.fit_transform(x_train, t_train)
  new_x_test = pca.transform(x_test)

  results = KNearestNeighbors(new_x_train, t_train, new_x_test, t_test)
  results['num_components'] = num_components
  return results

def LinearRegression(x_train, t_train, x_test, t_test):
  m = linear_model.LinearRegression()
  m.fit (x_train, t_train)
  p = m.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error,
    "average weight": sum(m.coef_)/len(m.coef_)
  }

def KNearestNeighbors(x_train, t_train, x_test, t_test):
  min_error = float("inf")
  for k in range(5, 15+1):
    weights = 'uniform'
    knn = neighbors.KNeighborsRegressor(k, weights)
    t_out = knn.fit(x_train, t_train).predict(x_test)
    error = computeError(t_out, t_test)
    if (error < min_error):
      min_error = error
      best_fit = (x_test, t_test, t_out, k)

  return {
    "error": computeError(best_fit[2], best_fit[1]),
    "best k": best_fit[3]
  }

def RidgeRegression(x_train, t_train, x_test, t_test):
  clf = linear_model.RidgeCV(alphas=[10**(-i) for i in range(20)])
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error,
    "Regularization term": clf.alpha_
  }

def DescisionTreeRegression(x_train, t_train, x_test, t_test):
  clf = tree.DecisionTreeRegressor()
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error
  }

def BayesianRidgeRegression(x_train, t_train, x_test, t_test):
  clf = linear_model.BayesianRidge()
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error,
    "average weight": sum(clf.coef_)/len(clf.coef_)
  }

def GaussianProcessRegression(x_train, t_train, x_test, t_test):
  x_train, t_train = deDupe(x_train, t_train)
  gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
  gp.fit(x_train, t_train)
  pred, sigma2_pred = gp.predict(x_test, eval_MSE=True)
  error = computeError(pred, t_test)
  return {
    "error": error,
    "sigma2": sum(sigma2_pred.tolist())/len(sigma2_pred.tolist())
  }

def AdaBoostRegression(x_train, t_train, x_test, t_test):
  clf = AdaBoostRegressor()
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error,
  }

def GradientBoostingRegression(x_train, t_train, x_test, t_test):
  clf = GradientBoostingRegressor()
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error,
  }

def SupportVectorRegression(x_train, t_train, x_test, t_test):
  clf = SVR(C=1.0, epsilon=0.2)
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  error = computeError(p, t_test)
  return {
    "error": error,
  }

def KMeansOnClusters(x_train, t_train, x_test, t_test):
# run KNearestNeighbors where the cluster_centers_ are used as x_train, and
# t_train is the average cluster t
  
  min_error = -1
  best_k = -1
  best_neighbor_k = -1
  max_clusters = min(50, len(x_test)+1)
  for k in range(2, max_clusters):
    estimator = KMeans(n_clusters=k, init='k-means++', n_init=10, n_jobs=-1)
    estimator.fit(x_train, t_train)
    
    # [k by num_features] array of cluster centers
    x_cluster = estimator.cluster_centers_
    # [1 by num_examples] array of cluster assignments
    x_train_labels = estimator.labels_
    
    # find the average cpg per cluster
    t_cluster = numpy.zeros(k)
    for i in range(0, k):
      
      # get the examples in the ith cluster
      examples_in_i = numpy.where(x_train_labels==i)[0]
      
      # take the average cpg of those examples
      total_cpg = 0;
      for j in range(0, len(examples_in_i)):
        total_cpg += t_train[examples_in_i[j]]
      t_cluster[i] = total_cpg/len(examples_in_i)

    # run KNearestNeighbor on the new data set
    results = KNearestNeighbors(x_cluster, t_cluster, x_test, t_test)
    if (min_error==-1 or results["error"] < min_error):
      min_error = results["error"]
      best_k = k
      best_neighbor_k = results["best k"]

  return {
    "error": min_error,
    "best k": best_k,
    "best neighbor k": best_neighbor_k,
  }

def KMeansPerCluster(x_train, t_train, x_test, t_test):
# run LocallyWeighted for each cluster in predict(x_test), where
#   (x_train, t_train) is all of the examples in that cluster
  min_error = -1
  best_k = -1
  max_clusters = min(50, len(x_test))
  for k in range(30, max_clusters+1):
    estimator = KMeans(n_clusters=k, init='k-means++', n_init=10)
    estimator.fit(x_train, t_train)
  
    # [k by num_features] array of cluster centers
    x_cluster = estimator.cluster_centers_
    # [1 by num_examples] array of cluster assignments
    x_train_labels = estimator.labels_
    x_test_labels = estimator.predict(x_test)
    
    # run KNearestNeighbors for each cluster, and average the error
    avg_error = 0
    for i in range(0, k):
      train_examples_in_i = numpy.where(x_train_labels==i)[0]
      test_examples_in_i = numpy.where(x_test_labels==i)[0]
      
      x_train_cluster = [x_train[x] for x in train_examples_in_i]
      t_train_cluster = [t_train[x] for x in train_examples_in_i]
      x_test_cluster = [x_test[x] for x in test_examples_in_i]
      t_test_cluster = [t_test[x] for x in test_examples_in_i]
      
      # only compute the KNearest if there are some examples in the new x_test
      if (len(x_test_cluster) > 0):
        results = KNearestNeighbors(x_train_cluster, t_train_cluster, x_test_cluster, t_test_cluster)
        avg_error += results["error"]
  
    # store the best cluster size
    if (min_error==-1 or (avg_error/k) < min_error):
      min_error = (avg_error/k)
      best_k = k
  return {
    "error": min_error,
    "best k": best_k,

  }

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
  """Ensure no x's repeat"""
  d = {}
  indices = [d.setdefault(str(x[i]), i) for i in range(len(x)) if str(x[i]) not in d]
  new_x = [x[i] for i in indices]
  new_t = [t[i] for i in indices]
  return new_x, new_t

def mergeResults(results):
  merged = {}
  for key in results[0]:
    mean = sum([result[key] for result in results])/len(results)
    stdev = (sum([(result[key] - mean)**2])/len(results))**0.5
    merged[key] = "{0} +- {1}".format("%.3f" % mean, "%.3f" % stdev)
  return merged

def learnAllUnlearnedModels():
  results_file = os.path.expanduser(private_consts.SAVE_DIR)+"results.json"
  try:
    with open(results_file) as f:
        results = json.loads(f.read())
  except:
    print "No results file. Starting from scratch."
    results = {}

  needToSave = False

  num_examples = [10, 30, 100, 300, 500, 1000]
  algorithms = [BayesianRidgeRegression, GaussianProcessRegression,
    GradientBoostingRegression, KMeansPerCluster]

  for n in num_examples:
    (x,t,vocabulary) = load_data(n)
    (x,t) = shuffle_data(x,t)
    for fn in algorithms:
      algorithm = fn.__name__
      if algorithm not in results:
        results[algorithm] = {}
      experiment_key = "{0} examples".format(n)
      if experiment_key not in results[algorithm]:
        print "Running {0} on {1} examples...".format(algorithm, n)
        kf = cross_validation.KFold(n, n_folds=5, indices=True)
        k_results = []
        for train, test in kf:
          x_train = [x[i] for i in train]
          t_train = [t[i] for i in train]
          x_test = [x[i] for i in test]
          t_test = [t[i] for i in test]
          start = time.time()
          result = fn(x_train, t_train, x_test, t_test)
          finish = time.time()
          result["time"] = finish - start
          k_results.append(result)
        results[algorithm][experiment_key] = mergeResults(k_results)
        needToSave = True

  if needToSave:
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
