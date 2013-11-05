"""
Take food items previously downloaded and process them into
usable examples.
"""

import private_consts
from load_save_data import save_data
from stemming.porter2 import stem
import string

import pickle
import os

def process_item(raw_item, food_vocabulary, stop_words):
  """
  Take an object downloaded from Nutritionix and turn it into an example.
  Return None if we could not process it.
  """
  tokens = extract_tokens(raw_item, stop_words)

  calories = raw_item["nf_calories"]
  grams = raw_item["nf_serving_weight_grams"]
  if not grams:
    # Just skip items without a weight
    return
  cpg = calories * 1.0 / grams; # Calories per gram

  return (tokens, cpg)

def extract_tokens(raw_item, stop_words):
  """Remove duplicates, remove punctuation, remove stop words, apply stemming"""
  name = raw_item["item_name"]
  description = raw_item["item_description"]

  raw_tokens = set()
  if name:
    raw_tokens = raw_tokens.union(name.split())
  if description:
    raw_tokens = raw_tokens.union(description.split())

  tokens = set()
  for raw_token in raw_tokens:
    # To lowercase ASCII
    raw_token = str(raw_token).lower()
    raw_token = raw_token.replace("&reg;", "")

    # Remove punctuation. TODO(wjbillin): Needs work.
    raw_token = raw_token.translate(string.maketrans("",""), string.punctuation)

    # Don't add it if it's a stop word.
    if raw_token in stop_words:
      #if raw_token in dummy_set:
      continue

    tokens.add(stem(raw_token))
  return tokens

print "Loading saved api data..."
raw_file = os.path.expanduser(private_consts.SAVE_DIR)+"raw_data.pickle"

raw = pickle.load( open( raw_file, "rb" ) )

stopwords_file = os.path.expanduser(private_consts.SAVE_DIR)+"stop_words.pickle"
stop_words = pickle.load( open( stopwords_file, "rb" ) )

print "Building vocabulary..."
food_vocabulary = set()
for item in raw:
  food_vocabulary = food_vocabulary.union(extract_tokens(item, stop_words))

food_vocabulary = sorted(food_vocabulary)
print food_vocabulary

print "Processing examples..."
examples = [nonNone for nonNone in [process_item(item, food_vocabulary, stop_words) for item in raw] if nonNone]

print "Saving",len(examples),"examples..."

save_data(food_vocabulary, examples)

f = open(os.path.expanduser(private_consts.SAVE_DIR)+"token_list.txt", 'w')
f.write("\n".join(food_vocabulary))
f.close()

