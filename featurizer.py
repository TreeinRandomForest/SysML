import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os, shutil

plt.ion()

def process_logs(PATH):
    data = []
    for f in os.listdir(PATH): #think this lists files
        features = process_file(PATH + '/' + f)
        data.append(features)

        
    data = pd.DataFrame(data)
    return data

def process_file(path):
    features = {}
    PCT_LIST = [10, 25, 50, 75, 90, 99]
    
    interrupt_cols = ['rx_bytes', 'rx_desc', 'tx_bytes', 'tx_desc']
    for c in interrupt_cols:
        pcts = np.percentiles(df[c], PCT_LIST)
        for p in pct_list:
            features[f'{c}_{p}'] = pcts[...]
        
    ms_cols = ['instructions', 'joules', 'llc_miss', 'cycles', 'ref_cycles']
    for c in ms_cols:
        pass

    #extract from filename
    features['itr'] = ...
    features['dvfs'] = ...

    features['lat_99'] = ...
    features['joules'] = ...
    
    return features

if __name__ == '__main__':
    data = process_logs(PATH)
    '''
    if too slow, we'll spawn multiple threads

    import multiprocessing as mp
    manager = mp.Manager()
    '''
