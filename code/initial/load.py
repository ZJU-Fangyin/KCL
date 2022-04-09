import torch
from torch import nn
import pickle
from utils import Triples
import os.path as op


def fill(name2id, loaded_id2name, emb, loaded_emb):
    for idx, name in loaded_id2name.items():
        real_idx = name2id.get(name, None)
        if real_idx is None:
            continue
        emb[real_idx] = loaded_emb[idx]
    return emb


def load(file_name, total_entity2id, total_relation2id, e_dim=128, r_dim=128) -> (torch.Tensor, torch.Tensor):
    loaded_dict = pickle.load(open(file_name, 'rb'))
    loaded_relation_emb, loaded_id2relation = loaded_dict['relation'], loaded_dict['id2relation']
    loaded_entity_emb, loaded_id2entity = loaded_dict['entity'], loaded_dict['id2entity']
    relation_emb = nn.Embedding(len(total_relation2id), r_dim).weight.data
    entity_emb = nn.Embedding(len(total_entity2id), e_dim).weight.data

    relation_emb = fill(total_relation2id, loaded_id2relation, relation_emb, loaded_relation_emb)
    entity_emb = fill(total_entity2id, loaded_id2entity, entity_emb, loaded_entity_emb)
    # import pdb; pdb.set_trace()
    with open(f'{op.basename(file_name[:-4])}_emb.pkl', 'wb') as f:
        dict_save = {
            'relation_emb': relation_emb,
            'entity_emb': entity_emb
        }
        pickle.dump(dict_save, f)
    return relation_emb, entity_emb

data = Triples()
relation_emb, entity_emb = load('RotatE_128_64.pkl', total_entity2id=data.entity2id, total_relation2id=data.relation2id, e_dim=128, r_dim=64)

# def main():
#     file_name = 'RotatE_128_64.pkl'
#     relation2id = {'electron_affinity': 0, 'electronegativity': 1, 'family': 2, 'ionization_energy': 3,
#                    'metallicity': 4, 'periodic': 5, 'kkk': 6}
#     entity2id = {'0': 0, '0.82': 1, '0.93': 2, '0.98': 3, '1': 4, '1.0': 5, '1.31': 6, '1.55': 7, '1.61': 8, '1.65': 9,
#                  '1.66': 10, '1.83': 11, '1.88': 12, '1.90': 13, '1.91': 14, '1008.4': 15, '1011.8': 16, '1086.5': 17,
#                  '112': 18, '1139.9': 19, '118.4': 20, '1251.2': 21, '1312': 22, '1313.9': 23, '133.6': 24,
#                  '1402.3': 25, '141': 26, '15.7': 27, '153.9': 28, '1681': 29, '195': 30, '2': 31, '2.18': 32,
#                  '2.19': 33, '2.2': 34, '2.37': 35, '2.55': 36, '2.58': 37, '2.66': 38, '2.96': 39, '200': 40,
#                  '295.2': 41, '3': 42, '3.04': 43, '3.16': 44, '3.44': 45, '3.98': 46, '324.6': 47, '328': 48,
#                  '349': 49, '4': 50, '418.8': 51, '42.5': 52, '48.4': 53, '495.8': 54, '5': 55, '52.8': 56, '520.2': 57,
#                  '577.5': 58, '589.8': 59, '59.6': 60, '63.7': 61, '64.3': 62, '652.9': 63, '7': 64, '717.3': 65,
#                  '72': 66, '72.8': 67, '737.1': 68, '737.7': 69, '745.5': 70, '760.4': 71, '762.5': 72, '78': 73,
#                  '786.5': 74, '906.4': 75, '941': 76, '947': 77, '999.6': 78, 'Al': 79, 'As': 80, 'Br': 81, 'C': 82,
#                  'Ca': 83, 'Cl': 84, 'Co': 85, 'Cr': 86, 'Cu': 87, 'F': 88, 'Fe': 89, 'H': 90, 'I': 91, 'K': 92,
#                  'Li': 93, 'Mg': 94, 'Mn': 95, 'N': 96, 'Na': 97, 'Ni': 98, 'O': 99, 'P': 100, 'S': 101, 'Se': 102,
#                  'Si': 103, 'Zn': 104, 'alkali_metal': 105, 'halogen': 106, 'lively_nonmetal': 107, 'metalloid': 108,
#                  'nitrogen': 109, 'oxygen': 110, 'poor_metal': 111, 'rare_earth_metal': 112, 'transition_metal': 113,
#                  'gogogogo': 114}
#     relation_emb, entity_emb = load(file_name, entity2id, relation2id, 128, 64)


# main()
