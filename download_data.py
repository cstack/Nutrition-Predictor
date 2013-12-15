"""
Download food items from Nutritionix and save the raw
examples as a pickle (http://docs.python.org/2/library/pickle.html)
for later processing
"""

from nutritionix import Nutritionix
import private_consts

import re
import pickle
import os
import sys

ITEMS_PER_API_CALL = 50

def create_filter(only_calories):
  filt = {
    "item_type":1,
    "nf_calories":{
      "gt":0
    }
  }
  
  if not only_calories:
    filt["nf_serving_weight_grams"] = { "gt":0 };

  return filt

def download(examples_to_download, offset, only_calories):

  save_file = create_new_file(examples_to_download, offset, only_calories)

  nix = Nutritionix(app_id=private_consts.NUTRITIONIX["app_id"],
    api_key=private_consts.NUTRITIONIX["api_key"])

  items = []
  num_calls = examples_to_download/ITEMS_PER_API_CALL
  for i in range(num_calls):
    print "Downloading items {0} through {1}".format(i*ITEMS_PER_API_CALL + offset, (i+1)*ITEMS_PER_API_CALL + offset)
    results = nix.search().nxql(
      filters = create_filter(only_calories),
      offset = i*ITEMS_PER_API_CALL + offset,
      limit = ITEMS_PER_API_CALL
    ).json()

    items += [hit["_source"] for hit in results["hits"]]

  # Save data
  print "Saving " + str(len(items)) + " items"
  pickle.dump( items, open( save_file, "wb" ) )

  return

def create_new_file(num, offset, only_calories):
  end = num + offset
  filename = os.path.expanduser(private_consts.SAVE_DIR) + "raw_data_" + str(offset) + "_" + str(end)
  if only_calories:
    filename = filename + "_cal_only"
  filename = filename + ".pickle"

  return filename

# Beginning of execution.
if __name__ == "__main__":
  usage_error = "\nUsage: python download_data.py 1000 # Download 1000 examples, or\n python download_data.py 1000 200 # Download 1000 examples, with an offset of 200\n -c to retrieve items that have listed calories, otherwise uses the default of listed calories and listed grams"

  num_to_download = 0
  offset = 0

  if len(sys.argv) == 2:
    download(int(sys.argv[1]), 0, False) # start at the beginning, calories and grams
  elif len(sys.argv) == 3:
    if sys.argv[1] == "-c":
      download(int(sys.argv[2]), 0, True) # start at the beginning, calories only
    else:
      download(int(sys.argv[1]), int(sys.argv[2]), False) # start at an offset, calories and grams
  elif len(sys.argv) == 4 and sys.argv[1] == "-c":
    download(int(sys.argv[2]), int(sys.argv[3]), True) # start at an offset, calories only
  else:
    print usage_error
    exit(1)

