"""
Learn a linear regression model
"""

from sklearn import linear_model, neighbors, tree

import os, pickle, numpy, sys
import private_consts
from load_save_data import load_data, load_and_split_data
from utilities import pretty_print_predictions
import json

def LinearRegression(num_examples = 100, percent_train = 0.8):
  ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, percent_train)
  m = linear_model.LinearRegression()
  m.fit (x_train, t_train)
  p = m.predict(x_test)
  pretty_print_predictions(x_test, t_test, p, num_examples)
  error = computeError(p, t_test)
  return {
    "error": error,
    "average weight": sum(m.coef_)/len(m.coef_)
  }

def KNearestNeighbors(num_examples = 100, percent_train = 0.8):
  min_error = 1000
  for k in range(5, 15+1):
    ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, percent_train)
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

def RidgeRegression(num_examples = 100):
  ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, 0.8)
  clf = linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100])
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  pretty_print_predictions(x_test, t_test, p, num_examples)
  print "Regularization term:",clf.alpha_
  error = computeError(p, t_test)
  return {
    "error": error,
    "Regularization term": clf.alpha_
  }

def DescisionTreeRegression(num_examples = 100):
  ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, 0.8)
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

def learnAllUnlearnedModels():
  results_file = os.path.expanduser(private_consts.SAVE_DIR)+"results.json"
  try:
    with open(results_file) as f:
        results = json.loads(f.read())
  except:
    print "No results file. Starting from scratch."
    results = {}

  needToSave = False
  num_examples = [10, 100, 1000]
  algorithms = [LinearRegression, KNearestNeighbors, RidgeRegression, DescisionTreeRegression]

  for fn in algorithms:
    algorithm = fn.__name__
    if algorithm not in results:
      results[algorithm] = {}
    for n in num_examples:
      if "{0} examples".format(n) not in results[algorithm]:
        print "Running {0} on {1} examples...".format(algorithm, n)
        result = fn(n)
        results[algorithm]["{0} examples".format(n)] = result
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
