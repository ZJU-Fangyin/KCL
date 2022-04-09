import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import RDLogger
import logging
logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*')
import lmdb
import numpy as np
from rdkit import Chem
import dgl
import torch
import torch.nn as nn
import pdb
from load_triples import Triples
from torch.nn import Embedding
from pandarallel import pandarallel


def bondtype_features(bond):
    bondtype_list=['SINGLE','DOUBLE','TRIPLE','AROMATIC']
    bond2emb = {}
    for idx, bt in enumerate(bondtype_list):
        bond2emb[bt]= idx
    fbond = bond2emb[str(bond.GetBondType())]
    return fbond

def smiles_2_kgdgl(smiles):
    data = Triples()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Invalid mol found')
        return None

    connected_atom_list = []
    for bond in mol.GetBonds():
        connected_atom_list.append(bond.GetBeginAtomIdx())
        connected_atom_list.append(bond.GetEndAtomIdx())
    
    connected_atom_list = sorted(list(set(connected_atom_list)))
    connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}
    atoms_feature = [0 for _ in range(len(connected_atom_list))]

    # get all node ids and relations
    begin_atoms = []
    end_entities = []
    rel_features = []
    for atom in mol.GetAtoms():
        node_index = atom.GetIdx()
        symbol = atom.GetSymbol()
        atomicnum = atom.GetAtomicNum()
        if node_index not in connected_atom_list:
            continue

        atoms_feature[connected_atom_map[node_index]] = atomicnum

        if symbol in data.entities:
            tid = [t for (r,t) in data.h2rt[data.entity2id[symbol]]]
            rid = [r for (r,t) in data.h2rt[data.entity2id[symbol]]]

            begin_atoms.extend([node_index]*len(tid))   # add head entities
            end_entities.extend(tid)    # add tail eneities
            rel_features.extend(i+4 for i in rid)

    # get list of tail entity ids and features
    if end_entities:
        entity_id = sorted(list(set(end_entities)))
        node_id = [i+len(connected_atom_list) for i in range(len(entity_id))]
        entid2nodeid = dict(zip(entity_id, node_id)) # dict: t_id in triples --> node_id in dglgraph
        nodeids = [entid2nodeid[i] for i in end_entities]   # list of tail entity id

        nodes_feature = [i+104 for i in entity_id]

    # get list of atom ids and bond features
    begin_indexes = []
    end_indexes = []
    bonds_feature = []
    
    for bond in mol.GetBonds():
        bond_feature = bondtype_features(bond)

        begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        bonds_feature.append(bond_feature)

        begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        bonds_feature.append(bond_feature)

    # add ids and features of tail entities and relations
    if begin_atoms:
        begin_indexes.extend(begin_atoms)
        end_indexes.extend(nodeids)
        atoms_feature.extend(nodes_feature)
        bonds_feature.extend(rel_features)

    
    # create dglgraph
    graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)
    graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)
    graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long) 
    return graph

def smiles_2_dgl(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Invalid mol found')
        return None

    connected_atom_list = []
    for bond in mol.GetBonds():
        connected_atom_list.append(bond.GetBeginAtomIdx())
        connected_atom_list.append(bond.GetEndAtomIdx())
    
    connected_atom_list = sorted(list(set(connected_atom_list)))
    connected_atom_map = {k: v for k, v in zip(connected_atom_list, list(range(len(connected_atom_list))))}
    atoms_feature = [0 for _ in range(len(connected_atom_list))]

    # get all node ids and relations
    for atom in mol.GetAtoms():
        node_index = atom.GetIdx()
        atomicnum = atom.GetAtomicNum()
        if node_index not in connected_atom_list:
            continue

        atoms_feature[connected_atom_map[node_index]] = atomicnum

    # get list of atom ids and bond features
    begin_indexes = []
    end_indexes = []
    bonds_feature = []
    
    for bond in mol.GetBonds():
        bond_feature = bondtype_features(bond)

        begin_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        bonds_feature.append(bond_feature)

        begin_indexes.append(connected_atom_map[bond.GetEndAtomIdx()])
        end_indexes.append(connected_atom_map[bond.GetBeginAtomIdx()])
        bonds_feature.append(bond_feature)
    
    # create dglgraph
    graph = dgl.graph((begin_indexes, end_indexes), idtype=torch.int32)
    graph.edata['e'] = torch.tensor(bonds_feature, dtype=torch.long)
    graph.ndata['h'] = torch.tensor(atoms_feature, dtype=torch.long) 
    return graph

if __name__ == '__main__':
    output_idx = 1
    data_path = f'./raw/zinc15_250K_2D.csv'

    pandarallel.initialize()
    data = pd.read_csv(data_path)
    # 创建数据库文件
    env = lmdb.open(f'./zinc15_250K_2D', map_size=int(1e12), max_dbs=2)
    # 创建对应的数据库
    graphs_db = env.open_db('graph'.encode())
    kgraphs_db = env.open_db('kgraph'.encode())

    graphs = data['smiles'].parallel_apply(smiles_2_dgl).to_list()
    graphs = list(filter(None, graphs))

    kgraphs = data['smiles'].parallel_apply(smiles_2_kgdgl).to_list()
    kgraphs = list(filter(None, kgraphs))
    # 把数据写入到LMDB中
    with env.begin(write=True) as txn:
        for idx, graph in tqdm(enumerate(graphs)):
            txn.put(str(idx).encode(), pickle.dumps(graph), db=graphs_db)
        for idx, kgraph in tqdm(enumerate(kgraphs)):
            txn.put(str(idx).encode(), pickle.dumps(kgraph), db=kgraphs_db)
    env.close()

    import random
    import pickle
    i = list(range(len(graphs)))
    random.shuffle(i)

    with open('zinc15_250K_2D.pkl', 'wb') as f:
        pickle.dump(i, f)

        
    # graph1 = smiles_2_kgdgl('C(C1CCCCC1)NN')
    # graph2 = smiles_2_dgl('C(C1CCCCC1)NN')
    # print(graph.edges())
    # print(graph.nodes())
