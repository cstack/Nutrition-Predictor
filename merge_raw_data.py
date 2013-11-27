import os
import pickle
import sys

import private_consts

usage_string = 'Usage: merge_raw_data.py infile_1[.pickle] infile_2[.pickle] outfile[.pickle]'

if not len(sys.argv) == 4:
  print(usage_string)

raw_file_1 = os.path.expanduser(private_consts.SAVE_DIR) + sys.argv[1] + '.pickle'
raw_file_2 = os.path.expanduser(private_consts.SAVE_DIR) + sys.argv[2] + '.pickle'

raw_data_1 = pickle.load( open( raw_file_1, "rb" ) )
raw_data_2 = pickle.load( open( raw_file_2, "rb" ) )

raw_data_merged = raw_data_1 + raw_data_2

outfile = os.path.expanduser(private_consts.SAVE_DIR) + sys.argv[3] + '.pickle'
pickle.dump( raw_data_merged, open( outfile, "wb" ) )
