"""
Download food items from Nutritionix and save the raw
examples as a pickle (http://docs.python.org/2/library/pickle.html)
for later processing
"""

from nutritionix import Nutritionix
import private_consts

import pickle
import os
import sys

ITEMS_PER_API_CALL = 50

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
        "item_type":1
      },
      offset = i*ITEMS_PER_API_CALL,
      limit = ITEMS_PER_API_CALL
    ).json()

    items += [hit["_source"] for hit in results["hits"]]

  # Save data
  pickle.dump( items, open( save_file, "wb" ) )

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: python download_data.py 1000 # Download 1000 examples"
    exit(1)
  num_to_download = int(sys.argv[1])
  download(num_to_download, private_consts.SAVE_DIR)
