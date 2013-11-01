import private_consts
import pickle
import os

from nltk.corpus import stopwords

print stopwords.words("english")

save_file = os.path.expanduser(private_consts.SAVE_DIR)+"stop_words.pickle"
pickle.dump( stopwords.words("english"), open( save_file, "wb" ) )