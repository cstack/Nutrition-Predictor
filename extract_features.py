"""
Take food items previously downloaded and process them into
usable examples.
"""

import private_consts

import pickle
import os
from sets import Set

def process_item(raw_item):
  """
  Take an object downloaded from Nutritionix and turn it into an example.
  Return None if we could not process it.
  """
  tokens = extract_tokens(raw_item)
  calories = raw_item["nf_calories"]
  grams = raw_item["nf_serving_weight_grams"]
  if not grams:
    # Just skip items without a weight
    return
  cpg = calories * 1.0 / grams; # Calories per gram

  return (tokens, cpg)

def extract_tokens(raw_item):
  """Remove duplicates.
  TODO: Clean punctuation, remove stop words, apply stemming"""
  name = raw_item["item_name"]
  description = raw_item["item_description"]
  tokens = Set()
  if name:
    tokens = tokens.union(name.split())
  if description:
    tokens = tokens.union(description.split())
  return tokens


raw_file = os.path.expanduser(private_consts.SAVE_DIR)+"raw_data.pickle"

raw = pickle.load( open( raw_file, "rb" ) )

for item in raw:
  example = process_item(item)
  if example:
    print example[1], example[0]
