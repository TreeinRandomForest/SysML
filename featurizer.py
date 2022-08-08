import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os, shutil

plt.ion()

def process_logs(PATH):
	data = []
	for f in os.listdir(PATH): #think this lists files
		print(f)
		features = process_file(PATH + '/' + f)
		data.append(features)

	data = pd.DataFrame(data)
	print("Data: ")
	print(data)
	print()
	print()

	return data

def process_file(path):
	features = {}
	values = pd.read_csv(path, sep = ' ', skiprows = 1, index_col = 0, names = ['rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp'])
	print(values.head())
#	print(values.shape)
#	print(values['rx_bytes'].head())

	PCT_LIST = [10, 25, 50, 75, 90, 99]
	
	interrupt_cols = ['rx_bytes', 'rx_desc', 'tx_bytes', 'tx_desc']
	for c in interrupt_cols:
		pcts = np.percentile(values[c], PCT_LIST)
#		print(c, "percentiles: ")
#		print(pcts)
		for i, p in enumerate(PCT_LIST):
			features[f'{c}_{p}'] = pcts[i]
	    
	ms_cols = ['instructions', 'joules', 'llc_miss', 'cycles', 'ref_cycles']
	for c in ms_cols:
		pcts = np.percentile(values[c], PCT_LIST)
		for i, p in enumerate(PCT_LIST):
			features[f'{c}_{p}'] = pcts[i]

	#extract from filename
#	features['itr'] = ...
#	features['dvfs'] = ...
#	
#	features['lat_99'] = ...
#	features['joules'] = ...
	return features

if __name__ == '__main__':
	data = process_logs("./logs_group_0")
	data.to_csv("./logs_0_percentiles.csv", sep = ' ')

'''
if too slow, we'll spawn multiple threads

import multiprocessing as mp
manager = mp.Manager()
'''
