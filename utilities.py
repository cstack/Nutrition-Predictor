from load_save_data import load_data

def pretty_print_predictions(x_test, t_test, t_out, data_size):
  (x, t, vocabulary) = load_data(data_size)

  for i in range(len(x_test)):
    print "Actual:",t_test[i],", Predicted:",t_out[i],[vocabulary[j] for j in range(len(x_test[i])) if (x_test[i][j] == 1)]


