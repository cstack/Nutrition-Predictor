"""
Learn a linear regression model
"""

from sklearn import linear_model

import os, pickle, numpy
import private_consts

clf = linear_model.LinearRegression()

data_file = os.path.expanduser(private_consts.SAVE_DIR)+"feature_data.pickle"

data = pickle.load( open( data_file, "rb" ) )

(x,t) = data

num_examples = len(x)
num_train = int(num_examples*0.8)
x_train = x[:num_train]
x_test = x[num_train:]
t_train = t[:num_train]
t_test = t[num_train:]

print "Learning..."
clf.fit (x_train, t_train)

print "Learned coefficients:", clf.coef_

print "Real values, predictions"
for i in range(len(x_test)):
  example = x_test[i]
  prediction = numpy.dot(example, clf.coef_)
  print t[i], prediction

