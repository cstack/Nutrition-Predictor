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
FILE_PATTERN = re.compile('\w*_\d_\d.pickle')

def download(examples_to_download, save_dir):
  save_file = os.path.expanduser(save_dir)+"raw_data.pickle"

  nix = Nutritionix(app_id=private_consts.NUTRITIONIX["app_id"],
    api_key=private_consts.NUTRITIONIX["api_key"])

  items = []
  num_calls = examples_to_download/ITEMS_PER_API_CALL
  for i in range(num_calls):
    print "Downloading items {0} through {1}".format(i*ITEMS_PER_API_CALL, (i+1)*ITEMS_PER_API_CALL)
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
      offset = i*ITEMS_PER_API_CALL,
      limit = ITEMS_PER_API_CALL
    ).json()

    items += [hit["_source"] for hit in results["hits"]]

  # Save data
  pickle.dump( items, open( save_file, "wb" ) )

if __name__ == "__main__":
  usage_error = "\nUsage: python download_data.py 1000 # Download 1000 " +
          "examples, or\n" +
          "python download_data.py raw_data_0_100.pickle 1000 # Download 100 " +
          "examples and append them to raw_data_0_100.pickle, forming a new file" +
          " raw_data_0_200.pickle"
  
  if len(sys.argv) < 2:
    print usage_error
    exit(1)

  num_to_download = 0
  if (FILE_PATTERN.match(sys.argv[1])):
    if len(sys.argv) != 3:
      print usage_error
      exit(1)
    num_to_download = int(sys.argv[2])
  else:
    num_to_download = int(sys.argv[1])
    download(num_to_download, private_consts.SAVE_DIR)
