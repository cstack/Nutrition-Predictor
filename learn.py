"""
Learn a linear regression model
"""

from sklearn import cross_validation, linear_model, neighbors, tree
from sklearn.decomposition import PCA

import os, pickle, numpy, sys, random
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
  pretty_print_predictions(x_test, t_test, p, num_examples)
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

  pretty_print_predictions(best_fit[0], best_fit[1], best_fit[2], num_examples)
  print "Best number of neighbors:",best_fit[3]
  return {
    "error": computeError(best_fit[2], best_fit[1]),
    "best k": best_fit[3]
  }

def RidgeRegression(x_train, t_train, x_test, t_test):
  clf = linear_model.RidgeCV(alphas=[10**(-i) for i in range(20)])
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  pretty_print_predictions(x_test, t_test, p, num_examples)
  print "Regularization term:",clf.alpha_
  error = computeError(p, t_test)
  return {
    "error": error,
    "Regularization term": clf.alpha_
  }

def DescisionTreeRegression(x_train, t_train, x_test, t_test):
  clf = tree.DecisionTreeRegressor()
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  pretty_print_predictions(x_test, t_test, p, num_examples)
  error = computeError(p, t_test)
  return {
    "error": error
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

def mergeResults(results):
  merged = {}
  for key in results[0]:
    mean = sum([result[key] for result in results])/len(results)
    stdev = (sum([(result[key] - mean)**2])/len(results))**0.5
    merged[key] = "{0} +- {1}".format(mean, stdev)
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

  num_examples = [10, 30, 100, 300]
  algorithms = [LinearRegression, KNearestNeighbors, RidgeRegression, DescisionTreeRegression, KNearestWithPCA]

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
          result = fn(x_train, t_train, x_test, t_test)
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
