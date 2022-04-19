import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from typing import List
from argparse import ArgumentParser, Namespace
import torch
from torchlight import set_seed, initialize_exp, snapshot
from dgllife.utils import EarlyStopping, Meter
from data import DataModule
from model import NonLinearPredictor, LinearPredictor, GCNEncoder
from model import GCNNodeEncoder, WeightedSumAndMax, MPNNGNN, Set2Set, KMPNNGNN
from torch.optim import Adam
# from analyse import pca_2d, tsne_2d, pattern_tsne_2d
import logging
# import matplotlib.pyplot as plt
import pdb
import pickle
logger = logging.getLogger()


class Reproduce(object):
    def __init__(self, args, data):
        self.args = args
        self.device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        self.data = data
        if self.args['data_name'] in ['FreeSolv', 'Lipophilicity', 'ESOL']:
            self.criterion = nn.MSELoss(reduction='none')
        elif self.args['data_name'] in ['BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'HIV', 'Tox21', 'MUV', 'PCBA']:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        if args['encoder_name'] == 'GNN':
            self.encoder = GCNNodeEncoder(args).to(self.device)
            self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
            self.readout = WeightedSumAndMax(self.encoder.out_dim).to(self.device)
            self.readout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))
        elif args['encoder_name'] == 'MPNN':  
            self.encoder = MPNNGNN(args).to(self.device)
            self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
            self.readout = Set2Set(self.encoder.out_dim, n_iters=6, n_layers=3).to(self.device)
            self.readout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))
        elif args['encoder_name'] == 'KMPNN':  
            self.loaded_dict = pickle.load(open(args['initial_path'], 'rb'))
            self.entity_emb, self.relation_emb = self.loaded_dict['entity_emb'], self.loaded_dict['relation_emb']
            self.encoder = KMPNNGNN(args, self.entity_emb, self.relation_emb).to(self.device)
            self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
            self.readout = Set2Set(self.encoder.out_dim, n_iters=6, n_layers=3).to(self.device)
            self.readout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))

        if args['predictor'] == 'nonlinear':
            self.predictor = NonLinearPredictor(self.readout.out_dim, data.task_num, self.args).to(self.device)
        elif args['predictor'] == 'linear':
            self.predictor = LinearPredictor(self.readout.out_dim, data.task_num, self.args).to(self.device)

        if args['eval'] == 'freeze':
            self.optimizer = Adam(self.predictor.parameters(), lr=self.args['lr'])
        else:
            self.optimizer = Adam([{"params": self.predictor.parameters()}, {"params": self.encoder.parameters()}, {"params": self.readout.parameters()}], lr=self.args['lr'])
       
    def run_train_epoch(self, dataloader):
        self.encoder.eval()
        self.predictor.train()
        total_loss = 0
        for batch_id, batch_data in enumerate(dataloader):
            smiles, bg, labels, masks = batch_data
            if len(smiles) == 1:
                continue
            bg, labels, masks = bg.to(self.device), labels.to(self.device), masks.to(self.device)

            with torch.no_grad():
                graph_embedding = self.readout(bg, self.encoder(bg))
            logits = self.predictor(graph_embedding)
            loss = (self.criterion(logits, labels) * (masks != 0).float()).mean()
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        logger.info(f'train loss: {total_loss}')

    def run_eval_epoch(self, dataloader):
        self.encoder.eval()
        self.predictor.eval()
        eval_meter = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(dataloader):
                smiles, bg, labels, masks = batch_data
                bg, labels = bg.to(self.device), labels.to(self.device)
                graph_embedding = self.readout(bg, self.encoder(bg))
                logits = self.predictor(graph_embedding)
                eval_meter.update(logits, labels, masks)
        return np.mean(eval_meter.compute_metric(self.data.matrix)) 

    def run(self):
        # self.run_analyse(self.data.train_dataloader(), f'{self.args["data_name"]}-{self.args["split_type"]}-nodemask-train.png')
        stopper = EarlyStopping(patience=args['patience'], metric=data.matrix)        
        for epoch_idx in range(self.args['epoch_num']):
            self.run_train_epoch(self.data.train_dataloader())
            val_score = self.run_eval_epoch(data.val_dataloader())
            test_score = self.run_eval_epoch(self.data.test_dataloader())
            logger.info(f'val score: {val_score} | test score: {test_score}')
            early_stop = stopper.step(val_score, self.predictor)
            if early_stop:
                break
        stopper.load_checkpoint(self.predictor)
        val_score = self.run_eval_epoch(data.val_dataloader())
        test_score = self.run_eval_epoch(data.test_dataloader())

        logger.info(f'val_score: {val_score} | test_score: {test_score}') 


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--data_name', type=str, default='SIDER')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--featurizer_type', type=str, default='random')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--split_type', type=str, default='random')
    parser.add_argument('--split_ratio', type=List, default=[0.8, 0.1, 0.1])
    
    parser.add_argument('--encoder_name', type=str, default='GNN')
    parser.add_argument('--encoder_path', type=str, default=None)
    parser.add_argument('--readout_path', type=str, default=None)
    parser.add_argument('--patience', type=int, default=50)

    parser.add_argument('--eval', type=str, default='freeze')
    parser.add_argument('--predictor', type=str, default='linear')
    parser.add_argument('--node_indim', type=int, default=128)
    parser.add_argument('--edge_indim', type=int, default=64)
    parser.add_argument('--hidden_feats', type=int, default=64)
    parser.add_argument('--node_hidden_feats', type=int, default=64)
    parser.add_argument('--edge_hidden_feats', type=int, default=128)
    parser.add_argument('--num_step_message_passing', type=int, default=6)    
    parser.add_argument('--gnn_norm', type=str, default=None) # None, both, right
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--batchnorm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)

    # predictor
    parser.add_argument('--predictor_dropout', type=float, default=0.0)
    parser.add_argument('--predictor_hidden_feats', type=int, default=64)

    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument("--dump_path", default="./dump", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_name", default="", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument('--initial_path', type=str, default='initial/RotatE_128_64_emb.pkl')

    return parser.parse_args().__dict__ 


if __name__ == '__main__':
    args = get_args()
    set_seed(args['seed'])

    logger, dump_folder = initialize_exp(Namespace(**args))
    args['dump_folder'] = dump_folder
    data = DataModule(args['encoder_name'], args['data_name'], args['num_workers'], args['split_type'], args['split_ratio'], args['batch_size'], args['seed'])

    reproducer = Reproduce(args, data)
    reproducer.run()
