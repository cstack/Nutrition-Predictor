"""
Take food items previously downloaded and process them into
usable examples.
"""

import private_consts
from stemming.porter2 import stem
import string

import pickle
import os
from sets import Set

def process_item(raw_item, food_vocabulary, stop_words):
  """
  Take an object downloaded from Nutritionix and turn it into an example.
  Return None if we could not process it.
  """
  
  tokens = extract_tokens(raw_item, stop_words)

  tokens_list = [0] * len(food_vocabulary)
  for token in tokens:
    tokens_list[food_vocabulary.index(token)] = 1
  
  calories = raw_item["nf_calories"]
  grams = raw_item["nf_serving_weight_grams"]
  cpg = calories * 1.0 / grams; # Calories per gram

  return (tokens_list, cpg)

def extract_tokens(raw_item, stop_words):
  """Remove duplicates, remove punctuation, remove stop words, apply stemming"""
  name = raw_item["item_name"]
  description = raw_item["item_description"]

  raw_tokens = Set()
  if name:
    raw_tokens = raw_tokens.union(name.split())
  if description:
    raw_tokens = raw_tokens.union(description.split())

  tokens = Set()
  for raw_token in raw_tokens:
    # To lowercase ASCII
    raw_token = str(raw_token).lower()
    raw_token = raw_token.replace("&reg;", "")

    # Remove punctuation. TODO(wjbillin): Needs work.
    raw_token = raw_token.translate(string.maketrans("",""), string.punctuation)

    # Don't add it if it's a stop word.
    if raw_token in stop_words:
      continue
    
    # Don't add the token if it is empty.
    if len(raw_token) == 0:
      continue

    # Don't add the token if it starts with a number.
    if raw_token[0].isdigit():
      continue

    tokens.add(stem(raw_token))
  return tokens

def get_word_frequency_diagnostics(phi_matrix):
  """Print diagnostic information about the vocabulary and how often
     the vocab words appear in our examples
  """
  if len(phi_matrix) == 0:
    return
  
  word_freq = {}
  # Initialize dict. TODO(wjbillin): There's probably a better way to do this.
  for i, val in enumerate(phi_matrix[0]):
    word_freq[i] = 0;

  for example in phi_matrix:
    for idx, count in enumerate(example)
      word_freq[idx] += count

  
  return


# Beginning of execution.
raw_file = os.path.expanduser(private_consts.SAVE_DIR)+"raw_data.pickle"

raw = pickle.load( open( raw_file, "rb" ) )

stopwords_file = os.path.expanduser(private_consts.SAVE_DIR)+"stop_words.pickle"
stop_words = pickle.load( open( stopwords_file, "rb" ) )

food_vocabulary = Set()
for item in raw:
  food_vocabulary = food_vocabulary.union(extract_tokens(item, stop_words))

food_vocabulary = sorted(food_vocabulary)

print food_vocabulary

x_data = []
t_data = []
for item in raw:
  example = process_item(item, food_vocabulary, stop_words)
  if example:
    x_data.append(example[0])
    t_data.append(example[1])

print "There are " + str(len(t_data)) + " items"
print "There are " + str(len(food_vocabulary)) + " distinct vocab words"

get_word_frequency_diagnostics(x_data)

save_file = os.path.expanduser(private_consts.SAVE_DIR)+"feature_data.pickle"
pickle.dump( (x_data, t_data) , open( save_file, "wb" ) )

f = open(os.path.expanduser(private_consts.SAVE_DIR)+"token_list.txt", 'w')
f.write("\n".join(food_vocabulary))
f.close()

