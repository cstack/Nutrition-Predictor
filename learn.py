"""
Learn a linear regression model
"""

from sklearn import linear_model

import os, pickle, numpy, sys
import private_consts
from load_save_data import load_data

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
  p = m.predict(x)
  error = computeError(p, t_test)
  return (m, error)

def crossValidationKNearestNeighbors(num_examples = 100, percent_train = 0.8, num_neighbors = 5):
  ((x_train, t_train), (x_test, t_test)) = load_and_split_data(num_examples, percent_train)
  weights = 'uniform'
  knn = neighbors.KNeighborsRegressor(num_neighbors, weights)
  t_out = knn.fit(x_train, t_train).predict(x_test)
  return computeError(t_out, t_test)  

def computeError(t_out, t_test):
  diff = [t_out[i] - t_test[i] for i in range(len(t_out))]
  error = sum([i**2 for i in diff]) / len(t_out)
  return error

def learn(num_examples=100):
  print "Loading {0} exmples...".format(num_examples)
  (x,t) = load_data(num_examples)

  print "Learning Linear Regression..."
  (model, error) = crossValidationLinearRegression(num_examples)

  print "Error per example:", error
  
  print "Learing K-Nearest Neighbors..."
  error = crossValidationKNearestNeighbors(num_examples)

  save_file = os.path.expanduser(private_consts.SAVE_DIR)+"model.pickle"
  pickle.dump( model , open( save_file, "wb" ) )
  print "Model saved in model.pickle"

if __name__ == "__main__":
  if len(sys.argv) < 2:
    num_examples = 100
  else:
    num_examples = int(sys.argv[1])

  learn(num_examples=num_examples)

