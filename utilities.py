from load_save_data import load_data
import private_consts
import os, json

def pretty_print_predictions(x_test, t_test, t_out, data_size):
  (x, t, vocabulary) = load_data(data_size)

  for i in range(len(x_test)):
    print "Actual:",t_test[i],", Predicted:",t_out[i],[vocabulary[j] for j in range(len(x_test[i])) if (x_test[i][j] == 1)]

def pretty_print_results():
  results_file = os.path.expanduser(private_consts.SAVE_DIR)+"results.json"
  try:
    with open(results_file) as f:
        results = json.loads(f.read())
  except:
    print "No results file."
    return
  print "Mean Squared Error on 300 examples:"
  for key in results:
    print key,":",results[key]["300 examples"]["error"]

def generate_data_sizes(max_size):
  sizes = []
  size = 10
  while size <= max_size:
    sizes.append(size)
    if (size*3) <= max_size:
      sizes.append(size*3)
    size *= 10
  return sizes
