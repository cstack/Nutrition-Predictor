"""
Learn a linear regression model
"""

from sklearn import linear_model, neighbors

import os, pickle, numpy, sys
import private_consts
from load_save_data import load_data, load_and_split_data
from utilities import pretty_print_predictions

def linearRegression(x, t):
  """Peform linear regression,
  return learned model m"""
  clf = linear_model.LinearRegression()
  num_examples = len(x)
  clf.fit (x, t)
  return clf

def crossValidationLinearRegression(num_examples = 100, percent_train = 0.8):
  ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, percent_train)
  m = linearRegression(x_train, t_train)
  p = m.predict(x_test)
  pretty_print_predictions(x_test, t_test, p, num_examples)
  error = computeError(p, t_test)
  return (m, error)

def crossValidationKNearestNeighbors(num_examples = 100, percent_train = 0.8, num_neighbors = 5):

  min_error = 1000
  for i in range(5, 15+1):
    ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, percent_train)
    weights = 'uniform'
    knn = neighbors.KNeighborsRegressor(i, weights)
    t_out = knn.fit(x_train, t_train).predict(x_test)
    error = computeError(t_out, t_test)
    if (error < min_error):
      min_error = error
      best_fit = (x_test, t_test, t_out, i)

  pretty_print_predictions(best_fit[0], best_fit[1], best_fit[2], num_examples)
  print "Best number of neighbors:",best_fit[3]
  return computeError(best_fit[2], best_fit[1])

def crossValidationRidgeRegression(num_examples = 100):
  ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, 0.8)
  clf = linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100])
  clf.fit(x_train, t_train)
  p = clf.predict(x_test)
  pretty_print_predictions(x_test, t_test, p, num_examples)
  print "Regularization term:",clf.alpha_
  error = computeError(p, t_test)
  return error

def computeError(t_out, t_test):
  diff = [t_out[i] - t_test[i] for i in range(len(t_out))]
  error = sum([i**2 for i in diff]) / len(t_out)
  return error

def learn(num_examples=100):
  print "Loading {0} exmples...".format(num_examples)

  #print "Learning Linear Regression..."
  #(model, error) = crossValidationLinearRegression(num_examples)
  #print "Linear Regression Mean Squared Error:", error

  #print "Learing K-Nearest Neighbors..."
  #error = crossValidationKNearestNeighbors(num_examples)
  #print "K-Nearest Neighbors Error:", error

  print "Learing Ridge Regression..."
  error = crossValidationRidgeRegression(num_examples)
  print "Ridge Regression Error:", error

if __name__ == "__main__":
  if len(sys.argv) < 2:
    num_examples = 100
  else:
    num_examples = int(sys.argv[1])

  learn(num_examples=num_examples)

