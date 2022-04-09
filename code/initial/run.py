import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import pickle
import random
import pdb
from tqdm import tqdm

from model import KGEModel
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from utils import Triples


class Runner:
    def __init__(self, params):
        self.params = params
        data = Triples()
        self.entity2id, self.relation2id = data.entity2id, data.relation2id
        self.train_triples = data.triples

        self.id2entity = {idx: ent for ent, idx in self.entity2id.items()}
        self.id2relation = {idx: ent for ent, idx in self.relation2id.items()}
        self.params.nentity = len(self.entity2id)
        self.params.nrelation = len(self.relation2id)
        print(f'{self.params.nentity} entities, {self.params.nrelation} relations')

        self.kge_model = KGEModel(
            model_name=self.params.model,
            nentity=self.params.nentity,
            nrelation=self.params.nrelation,
            hidden_dim=self.params.hidden_dim,
            gamma=self.params.gamma,
            double_entity_embedding=self.params.double_entity_embedding,
            double_relation_embedding=self.params.double_relation_embedding,
        )
        # pdb.set_trace()
        if self.params.cuda:
            self.kge_model = self.kge_model.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kge_model.parameters()),
            lr=self.params.learning_rate
        )

        self.train_iterator = self.get_train_iter()

    def run(self):
        best_result_dict = dict()
        for step in range(self.params.max_steps):
            training_logs = []
            log = self.kge_model.train_step(self.kge_model,
                                            self.optimizer,
                                            self.train_iterator,
                                            self.params)
            # training_logs.append(log)
            # print(log)
            print(f"[{step}] Loss={log['loss']:.5f}")
        self.save()

    def get_train_iter(self):
        train_dataloader_head = DataLoader(
            TrainDataset(self.train_triples, self.params.nentity, self.params.nrelation,
                         self.params.negative_sample_size, 'head-batch'),
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=max(1, self.params.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(self.train_triples, self.params.nentity, self.params.nrelation,
                         self.params.negative_sample_size, 'tail-batch'),
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=max(1, self.params.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        return train_iterator

    def save(self):
        with open(f'{self.params.model}_{self.kge_model.entity_dim}_{self.kge_model.relation_dim}.pkl', 'wb') as f:
            dict_save = {
                'id2entity': self.id2entity,
                'id2relation': self.id2relation,
                'entity': self.kge_model.entity_embedding.data,
                'relation': self.kge_model.relation_embedding.data
            }
            pickle.dump(dict_save, f)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--random_seed', default=1234, type=int)
    parser.add_argument('--model', default='RotatE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=32, type=int)
    parser.add_argument('-d', '--hidden_dim', default=64, type=int)
    parser.add_argument('-g', '--gamma', default=19.9, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-r', '--regularization', default=1e-9, type=float)
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.025, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=2, type=int, help='train log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)


def main():
    params = parse_args()
    result_dict_list = []

    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    runner = Runner(params)
    runner.run()


'''
CUDA_VISIBLE_DEVICES=1 python run.py -adv -de
'''
main()
#
# params = parse_args()
# params.fold=0
# runner = Runner(params)
# runner.evaluate()
