import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.family':'Times New Roman', 'font.size': 15})

for filer,setup in [('00112','0 0 1 1 2 3'), ('30112','3 0 1 1 2 3'), ('32000','3 2 0 0 0 3'), ('32102','3 2 1 0 2 3'), ('32112','3 2 1 1 2 3')]:
  data_params = setup.split(' ')
  n1 = int(data_params[0]) #n1 num sys inputs
  n2 = int(data_params[1]) #n2 num sys outputs
  k1 = int(data_params[2]) #k1 num nonsys reps input
  k2 = int(data_params[3]) #k2 num nonsys reps output
  r =  float(data_params[4]) #r scale

  sys_train_mean = np.mean(np.genfromtxt(filer+'/train_accs_sys.txt'), axis=1)
  non_train_mean = np.mean(np.genfromtxt(filer+'/train_accs_non.txt'), axis=1)
  sys_train_std = np.std(np.genfromtxt(filer+'/train_accs_sys.txt'), axis=1)
  non_train_std = np.std(np.genfromtxt(filer+'/train_accs_non.txt'), axis=1)
  sys_test_mean = np.mean(np.genfromtxt(filer+'/test_accs_sys.txt'), axis=1)
  non_test_mean = np.mean(np.genfromtxt(filer+'/test_accs_non.txt'), axis=1)
  sys_test_std = np.std(np.genfromtxt(filer+'/test_accs_sys.txt'), axis=1)
  non_test_std = np.std(np.genfromtxt(filer+'/test_accs_non.txt'), axis=1)

  epochs = 30
  if k1 == 0 and k2 == 0:
      epochs = 60

  if n2 > 0:
    plt.plot(sys_train_mean[:epochs], color='red', label=r'Compositional Training Loss')
    plt.fill_between(np.arange(epochs), sys_train_mean[:epochs] - sys_train_std[:epochs], sys_train_mean[:epochs] + sys_train_std[:epochs], alpha=0.5, color='red')
  if k2 > 0:
    plt.plot(non_train_mean[:epochs], color='green', label=r'Non-compositional Training Loss')
    plt.fill_between(np.arange(epochs), non_train_mean[:epochs] - non_train_std[:epochs], non_train_mean[:epochs] + non_train_std[:epochs], alpha=0.5, color='green')
  #plt.ylim([-0.05, np.max([sys_sys_norms,non_sys_norms,sys_non_norms,non_non_norms])+0.55])
  #plt.xlim([-10, num_epochs])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Quadratic Loss")
  plt.xlabel("Epoch number")
  plt.legend() #loc='upper left', bbox_to_anchor=(0.02, 0.52, 0.5, 0.5), ncol=2)
  plt.grid()
  plt.savefig(filer+'/train_losses.pdf')
  plt.close()

  if n2 > 0:
    plt.plot(sys_test_mean[:epochs], color='red', label=r'Compositional Test Loss')
    plt.fill_between(np.arange(epochs), sys_test_mean[:epochs] - sys_test_std[:epochs], sys_test_mean[:epochs] + sys_test_std[:epochs], alpha=0.5, color='red')
  if k2 > 0:
    plt.plot(non_test_mean[:epochs], color='green', label=r'Non-compositional Test Loss')
    plt.fill_between(np.arange(epochs), non_test_mean[:epochs] - non_test_std[:epochs], non_test_mean[:epochs] + non_test_std[:epochs], alpha=0.5, color='green')
  #plt.ylim([-0.05, np.max([sys_sys_norms,non_sys_norms,sys_non_norms,non_non_norms])+0.55])
  #plt.xlim([-10, num_epochs])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Quadratic Loss")
  plt.xlabel("Epoch number")
  plt.legend() #loc='upper left', bbox_to_anchor=(0.02, 0.52, 0.5, 0.5), ncol=2)
  plt.grid()
  plt.savefig(filer+'/test_losses.pdf')
  plt.close()
