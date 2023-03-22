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

if __name__ == '__main__':
    data_path = f'./raw/zinc15_250K_2D.csv'
    # data_path = 'raw/bbbp.csv'
    begin = int(sys.argv[1:][0])
    data = pd.read_csv(data_path)
    smiles_list = data['smiles'].to_list()[begin:begin + 50000]
    # mols
    mols = []
    for ind, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(mol)
        mols.append(mol)
    # fingerprints
    fps = []
    for x in tqdm(mols):
        fps.append(AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024))
    dists = []
    nfps = len(fps)
    dists_list = []
    for i in tqdm(range(nfps)):
        if i == 0:
            continue
        #         # for i in tqdm(range(nfps - 1, 0, -1)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists_i = [(1 - x) for x in sims]
        dists_list.extend(dists_i)
    with open(f'dists_split_5w/{begin}.pkl', 'wb') as f:
        pickle.dump(dists_list, f)
    # scaffold_sets = Butina.ClusterData(dists_list, nfps, distThresh=0.3, isDistData=True)
    # scaffold_sets = sorted(scaffold_sets, key=lambda x: -len(x))
    # pdb.set_trace()
    # txn.put(str(i).encode(), pickle.dumps(dists_i), db=dists_db)
