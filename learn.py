"""
Learn a linear regression model
"""

from sklearn import linear_model

import os, pickle, numpy
import private_consts
from load_save_data import load_data

def linearRegression(x, t):
  """Peform linear regression,
  return learned model m"""
  clf = linear_model.LinearRegression()
  num_examples = len(x)
  clf.fit (x, t)
  return clf

def predict(x, m):
  """Use learned model m to predict t values for x"""
  return m.predict(x)

def crossValidation(x,t):
  num_examples = len(x)
  num_train = int(num_examples*0.8)
  x_train = x[:num_train]
  x_test = x[num_train:]
  t_train = t[:num_train]
  t_test = t[num_train:]
  m = linearRegression(x_train, t_train)
  p = predict(x_test, m)
  diff = [p[i] - t_test[i] for i in range(len(p))]
  error = sum([i**2 for i in diff]) / len(p)
  return (m, error)

print "Loading data..."
(x,t) = load_data()

print "Learning..."
(model, error) = crossValidation(x,t)

print "Error:", error

save_file = os.path.expanduser(private_consts.SAVE_DIR)+"model.pickle"
pickle.dump( model , open( save_file, "wb" ) )
print "Model saved in model.pickle"

