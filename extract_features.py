"""
Take food items previously downloaded and process them into
usable examples.
"""

import private_consts, utilities
from load_save_data import save_data
from stemming.porter2 import stem
import string

import pickle
import os
import re # reg ex
import sys

def process_item(raw_item, stop_words, cal_only):
  """
  Take an object downloaded from Nutritionix and turn it into an example.
  Return None if we could not process it.
  """

  tokens = extract_tokens(raw_item, stop_words)
  if len(tokens) == 0:
    return None

  calories = raw_item["nf_calories"]
  grams = raw_item["nf_serving_weight_grams"]

  if not cal_only:
    # Use cpg as our target metric.
    cpg = calories * 1.0 / grams

    if (cpg > 10):
      print "Ignoring cpg outlier:",cpg,tokens
      return None

    return (tokens, cpg)

  else:
    # Use calories as our target metric.
    return (tokens, calories)

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
      continue

    # Don't add the token if it is empty.
    if len(raw_token) == 0:
      continue

    # Don't add the token if it starts with a number.
    number_regex = re.compile('\w*\d+\w*')
    if number_regex.match(raw_token):
      continue

    tokens.add(stem(raw_token))
  return tokens

def prune_vocabulary(examples, food_vocabulary):
  """Print diagnostic information about the vocabulary and how often
     the vocab words appear in our examples. Remove rare tokens from
     the vocabulary and our examples.
  """
  if len(examples) == 0:
    return

  word_freq = {}
  # Initialize dict with all the tokens and a count of 0.
  for token in food_vocabulary:
    word_freq[token] = 0

  # For each time we see a term appear, increment the frequency.
  for example in examples:
    for token in example[0]:
      word_freq[token] = word_freq[token] + 1

  # Remove terms that only appear very rarely.
  min_count = 2 # The minimum num of times a term must appear to be in the vocab
  frequencies = set()
  for key in word_freq:
    if word_freq[key] > min_count:
      frequencies.add((word_freq[key], key))
    else:
      print "Removing word " + key
      food_vocabulary.remove(key)

  print "Pruned vocab from " + str(len(word_freq)) + " to " + str(len(frequencies))

  # Construct the set of examples to no longer include removed vocab words.
  pruned_examples = list()
  while len(examples) > 0:
    example = examples.pop()
    pruned_feature = example[0] & food_vocabulary
    # Add the new example to our pruned examples if it still has valid tokens.
    if len(pruned_feature) > 0:
      pruned_examples.append((pruned_feature, example[1]))
    else:
      print "Removing example " + str(example[0])
    # Remove the example from examples.

  # Update the examples set.
  examples.extend(pruned_examples)
  print "Pruned examples to " + str(len(examples))

  top_frequencies = sorted(frequencies)

  print "There are " + str(len(pruned_examples)) + " examples"
  print "There are " + str(len(food_vocabulary)) + " vocab words"
  print "Most common words:",",".join([str(i) for i in top_frequencies[-10:]])

  return

def build_vocabulary(examples):
  """Compile the set of words across all examples"""
  vocabulary = set()
  for example in examples:
    vocabulary = vocabulary.union(example[0])
  # TODO: This doesn't need to be sorted, right?
  return vocabulary


# Beginning of execution.
print "Loading saved api data..."

# Pull out the file argument, if there is one.
raw_file = os.path.expanduser(private_consts.SAVE_DIR)+"raw_data_total.pickle"
if len(sys.argv) == 2 and sys.argv[1] != "-c":
  raw_file = os.path.expanduser(private_consts.SAVE_DIR) + sys.argv[1]
elif len(sys.argv) == 3 and sys.argv[1] == "-c":
  raw_file = os.path.expanduser(private_consts.SAVE_DIR) + sys.argv[2]
raw = pickle.load( open( raw_file, "rb" ) )

# Pull out stopwords from pickle
stopwords_file = os.path.expanduser(private_consts.SAVE_DIR)+"stop_words.pickle"
stop_words = pickle.load( open( stopwords_file, "rb" ) )

print "Processing examples..."
cal_only = True if (len(sys.argv) > 1 and sys.argv[1] == "-c") else False
examples = [nonNone for nonNone in [process_item(item, stop_words, cal_only) for item in raw] if nonNone]
vocabulary = build_vocabulary(examples)

# TODO: Make one more pass through vocab to throw out rare words.
prune_vocabulary(examples, vocabulary)
# Then, make another pass through examples to remove words that are
# no longer part of the vocabulary.

data_sizes = utilities.generate_data_sizes(len(examples))
for num_examples in data_sizes:
  print "Saving",num_examples,"examples..."
  save_data(vocabulary, examples[:num_examples])
print "Saving",len(examples),"examples"
save_data(vocabulary, examples)

f = open(os.path.expanduser(private_consts.SAVE_DIR)+"human_readable_data.txt", 'w')
f.write("\n".join([str(example[1])+", "+str(sorted(example[0])) for example in examples]))
f.close()

