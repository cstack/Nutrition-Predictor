import private_consts

import os
import pickle

def save_data(vocabulary, examples):
  save_file = os.path.expanduser(private_consts.SAVE_DIR)+"uninflated_data.{0}.pickle".format(len(examples))
  pickle.dump( {"vocabulary":vocabulary, "examples":examples} , open( save_file, "wb" ) )

def load_data(num_examples = 100):
  save_file = os.path.expanduser(private_consts.SAVE_DIR)+"uninflated_data.{0}.pickle".format(num_examples)
  uninflated_data = pickle.load( open( save_file, "rb" ) )

  vocabulary = uninflated_data["vocabulary"]

  x = []
  t = [example[1] for example in uninflated_data["examples"]]
  for example in uninflated_data["examples"]:
    tokens_list = [0] * len(vocabulary)
    for token in example[0]:
      tokens_list[vocabulary.index(token)] = 1
    x.append(tokens_list)

  return (x,t)

def load_and_split_data(num_examples = 100, percent_train = 0.8):
  (x, t) = load_data(num_examples)
  num_train = int(num_examples*percent_train)

  x_train = x[:num_train]
  x_test = x[num_train:]
  t_train = t[:num_train]
  t_test = t[num_train:]

  return ((x_train, t_train), (x_test, t_test))
