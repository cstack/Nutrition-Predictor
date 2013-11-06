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

def download(examples_to_download, offset, save_file):

  nix = Nutritionix(app_id=private_consts.NUTRITIONIX["app_id"],
    api_key=private_consts.NUTRITIONIX["api_key"])

  items = []
  num_calls = examples_to_download/ITEMS_PER_API_CALL
  for i in range(num_calls):
    print "Downloading items {0} through {1}".format(i*ITEMS_PER_API_CALL + offset, (i+1)*ITEMS_PER_API_CALL + offset)
    results = nix.search().nxql(
      filters = {
        "item_type":1,
        "nf_serving_weight_grams":{
          "gt":0
        },
        "nf_calories":{
          "gt":0
        }
      },
      offset = i*ITEMS_PER_API_CALL + offset,
      limit = ITEMS_PER_API_CALL
    ).json()

    items += [hit["_source"] for hit in results["hits"]]

  # Save data
  pickle.dump( items, open( save_file, "wb" ) )

  return

def create_new_file(num, offset):
  end = int(num) + int(offset);
  return os.path.expanduser(private_consts.SAVE_DIR) + "raw_data_" + offset + "_" + str(end) + ".pickle"

# Beginning of execution.
if __name__ == "__main__":
  usage_error = "\nUsage: python download_data.py 1000 # Download 1000 examples, or\n python download_data.py 1000 200 # Download 1000 examples, with an offset of 200"
  
  if len(sys.argv) < 2 or len(sys.argv) > 3:
    print usage_error
    exit(1)

  num_to_download = 0
  offset = 0
  if len(sys.argv) == 3:
    save_file = create_new_file(sys.argv[1], sys.argv[2])
    download(int(sys.argv[1]), int(sys.argv[2]), save_file)

  else:
    save_file = create_new_file(sys.argv[1], "0")
    download(int(sys.argv[1]), 0, save_file)
