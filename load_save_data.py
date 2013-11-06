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
