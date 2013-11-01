"""
Take food items previously downloaded and process them into
usable examples.
"""

import private_consts
from stemming.porter2 import stem
import string
from nltk.corpus import stopwords

import pickle
import os
from sets import Set

def process_item(raw_item, food_vocabulary):
  """
  Take an object downloaded from Nutritionix and turn it into an example.
  Return None if we could not process it.
  """
  tokens = extract_tokens(raw_item)
  
  tokens_list = [0] * len(food_vocabulary)
  for token in tokens:
    tokens_list[food_vocabulary.index(token)] = 1
  
  calories = raw_item["nf_calories"]
  grams = raw_item["nf_serving_weight_grams"]
  if not grams:
    # Just skip items without a weight
    return
  cpg = calories * 1.0 / grams; # Calories per gram

  return (tokens_list, cpg)

def extract_tokens(raw_item):
  """Remove duplicates.
  TODO: Clean punctuation, remove stop words, apply stemming"""
  name = raw_item["item_name"]
  description = raw_item["item_description"]

  raw_tokens = Set()
  if name:
    raw_tokens = raw_tokens.union(name.split())
  if description:
    raw_tokens = raw_tokens.union(description.split())

  # remove the stop words

  final_tokens = Set()
  for token in tokens:
    if token not in stopwords.words("english"):
      final_tokens.add(token)

  return final_tokens

  tokens = Set()
  for raw_token in raw_tokens:
    # To ASCII
    raw_token = str(raw_token)
    raw_token = raw_token.translate(string.maketrans("",""), string.punctuation)
    tokens.add(stem(raw_token.lower()))
  return tokens

raw_file = os.path.expanduser(private_consts.SAVE_DIR)+"raw_data.pickle"

raw = pickle.load( open( raw_file, "rb" ) )

food_vocabulary = Set()
for item in raw:
  food_vocabulary = food_vocabulary.union(extract_tokens(item))

food_vocabulary = sorted(food_vocabulary)
print food_vocabulary

print stopwords.words("english")

x_data = []
t_data = []
for item in raw:
  example = process_item(item, food_vocabulary)
  if example:
    x_data.append(example[0])
    t_data.append(example[1])

save_file = os.path.expanduser(private_consts.SAVE_DIR)+"feature_data.pickle"
pickle.dump( (x_data, t_data) , open( save_file, "wb" ) )

