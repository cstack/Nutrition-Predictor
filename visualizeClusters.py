from load_save_data import *
from learn import deDupe, shuffle_data, computeError
from sklearn import gaussian_process, linear_model, neighbors
from sklearn.cluster import KMeans
import numpy
import random, time

def print_clusters(model, vocabulary):
  assert type(vocabulary) == list
  print "%d Clusters\n" % len(model.cluster_centers_)
  for i, cluster_center in enumerate(model.cluster_centers_):
    print_cluster(cluster_center, vocabulary)


def print_cluster(cluster_center, vocabulary):
  assert type(vocabulary) == list
  words, values = [], []
  for i, word_value in enumerate(cluster_center):
    if word_value >= 0.1:
      word = vocabulary[i]
      words.append(word)
      values.append(word_value)
  print "(%s)" % ", ".join(["%s (%.2f)" % (words[i], values[i]) for i in range(len(words))])

def print_example(example, vocabulary):
  assert type(vocabulary) == list
  words = []
  for i in range(len(example)):
    if example[i] > 0:
      words.append(vocabulary[i])
  print ", ".join(words)

def examples_in_cluster(x, t, cluster, labels):
  indices = [i for i in range(len(x)) if labels[i] == cluster]
  x_in_cluster = [x[i] for i in indices]
  t_in_cluster = [t[i] for i in indices]
  return x_in_cluster, t_in_cluster

NUM_EXAMPLES = 10000
PERCENT_TRAIN = 0.8
NUM_CLUSTERS = 100

random.seed("Bologna")
num_train = int(NUM_EXAMPLES * PERCENT_TRAIN)
(x,t,vocabulary_set) = load_data(NUM_EXAMPLES)
vocabulary = list(vocabulary_set)
x_train, t_train, x_test, t_test = split_data(x,t, percent_train=PERCENT_TRAIN)

try:
  print "Loading KMeans model"
  labeler = load_model("KMeansPerCluster", num_train)
except:
  print "Model does not exist yet. Running KMeans."
  labeler = KMeans(n_clusters=NUM_CLUSTERS, random_state=random.randint(0,9999999))
  labeler.fit(x_train, t_train)
  save_model(labeler, "KMeansPerCluster", num_train)

print "Labeling data with clusters"
train_labels = labeler.predict(x_train)
test_labels = labeler.predict(x_test)

predictors = []
try:
  print "Loading KNN models for each cluster."
  for i in range(NUM_CLUSTERS):
    predictors.append(load_model("KMeansPerCluster.cluster%d" % i, num_train))
except:
  print "Models do not exist yet. Running KNN."
  for i in range(NUM_CLUSTERS):
    print "Training predictor for cluster %d" % i
    predictor = neighbors.KNeighborsRegressor(10)
    x_in_cluster, t_in_cluster = examples_in_cluster(x_train, t_train, i, train_labels)
    if len(x_in_cluster):
      predictor.fit(x_in_cluster, t_in_cluster)
    predictors.append(predictor)
    save_model(predictor, "KMeansPerCluster.cluster%d" % i, num_train)

for i, example in enumerate(x_test):
  print "Looking at example"
  print_example(example, vocabulary)
  label = test_labels[i]
  print "It belongs to cluster"
  print_cluster(labeler.cluster_centers_[label], vocabulary)
  print "KNN predicts %f cpg" % predictors[label].predict([example])[0]
  print "It actually has %f cpg" % t_test[i]
  print "\n"

error = 0
for cluster in range(NUM_CLUSTERS):
  x_in_cluster, t_in_cluster = examples_in_cluster(x_test, t_test, cluster, test_labels)
  if len(x_in_cluster) == 0:
    continue
  predictions = predictors[cluster].predict(x_in_cluster)
  error += sum([(predictions[j] - t_test[j])**2 for j in range(len(predictions))])
error /= len(t_test)

print "MSE:", error
