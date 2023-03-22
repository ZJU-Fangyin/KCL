import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import RDLogger
import logging

import lmdb
import numpy as np
import dgl
import torch
import torch.nn as nn
import pdb
from torch.nn import Embedding
from pandarallel import pandarallel
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
import sys

logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':
    # distThresh = 0.5
    # env = lmdb.open(f'./zinc_dists', map_size=int(1e12), max_dbs=1, readonly=True)
    # db = env.open_db('dists'.encode())
    # dists_list = []
    # with env.begin() as txn:
    #     total_num = txn.stat(db=db)['entries']
    #     for i in tqdm(range(1, total_num + 1)):
    #         # pdb.set_trace()
    #         dists_list.extend(pickle.loads(txn.get(str(i).encode(), db=db)))
    # env.close()
    # print('begin to cluster.')
    # scaffold_sets = Butina.ClusterData(dists_list, total_num, distThresh=distThresh, isDistData=True)
    # scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))
    # pdb.set_trace()
    thresh = 0.85
    begin = int(sys.argv[1:][0])
    with open(f'dists_split_5w/{begin}.pkl', 'rb') as f:
        dists_list = pickle.load(f)
    scaffold_sets = Butina.ClusterData(dists_list, 50000, distThresh=thresh, isDistData=True)
    scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))
    print(len(scaffold_sets))
    # pdb.set_trace()
    sorted_list = []
    for t in scaffold_sets:
        sorted_list.extend(list(t))
    sorted_list = list(map(lambda x: x + begin, sorted_list))
    with open(f'cluster_result/{begin}_{thresh}.pkl', 'wb') as f:
        pickle.dump(sorted_list, f)
