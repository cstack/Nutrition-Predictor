"""
Learn a linear regression model
"""

from sklearn import linear_model

clf = linear_model.LinearRegression()

"""
Below is an example.
TODO: make our data look like this, then run
linear regression on it.
"""

x = [[0, 0], [1, 1], [2, 2]]
t = [0, 1, 2]

print "Examples: ", x
print "Values:", t

print "Learning..."
clf.fit (x, t)

print "Learned coefficients:", clf.coef_
