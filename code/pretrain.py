from argparse import Namespace
from argparse import ArgumentParser
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import lmdb
import json
import pickle
import dgl
from data import LmdbDataModule
from model import GCNNodeEncoder, ContrastiveLoss, NonLinearProjector, WeightedSumAndMax, MPNNGNN, Set2Set, KMPNNGNN
from model import BernoulliDropoutNoisePNGenerator, GaussTimeNoisePNGenerator, GaussPlusNoisePNGenerator, NodeDropoutNoisePNGenerator, NodeMaskNoisePNGenerator
from torchlight import initialize_exp, set_seed, snapshot
from tensorboardX import SummaryWriter
import pdb
import logging
logger = logging.getLogger()


class Pretrainer(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        args['device'] = self.device
        self.global_step = 0
        self.epoch_num = args['epoch_num']

        # Data Utils
        self.data = LmdbDataModule(args['data_dir'], args['data_name'], args['batch_size'])
        self.env = lmdb.open(args['lmdb_env'], map_size=int(1e12), max_dbs=2, readonly=True)
        self.graph_db = self.env.open_db('graph'.encode())
        self.kgraph_db = self.env.open_db('kgraph'.encode())

        self.loaded_dict = pickle.load(open(args['initial_path'], 'rb'))
        self.entity_emb, self.relation_emb = self.loaded_dict['entity_emb'], self.loaded_dict['relation_emb']
        # Model Utils
        # if args['encoder_name'] == 'GNN':
        self.encoder = GCNNodeEncoder(args).to(self.device)
        self.readout = WeightedSumAndMax(self.encoder.out_dim).to(self.device)
        # elif args['encoder_name'] == 'MPNN':
        self.kencoder = KMPNNGNN(args, self.entity_emb, self.relation_emb).to(self.device)
        self.kreadout = Set2Set(self.kencoder.out_dim, n_iters=6, n_layers=3).to(self.device)

        if args['encoder_path'] != None:
            self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
        if args['readout_path'] != None:
            self.readout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))
        if args['kencoder_path'] != None:
            self.kencoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
        if args['kreadout_path'] != None:
            self.kreadout.load_state_dict(torch.load(args['readout_path'], map_location=self.device))

        self.projector = NonLinearProjector(self.readout.out_dim).to(self.device)
        self.optimizer = Adam([{"params": self.encoder.parameters()},{"params": self.kencoder.parameters()}, {"params": self.projector.parameters()}], lr=args['lr'])
        self.scheduler = ExponentialLR(self.optimizer, 0.99, -1)
        self.step_per_schedule = args['step_per_schedule']

        self.criterion = ContrastiveLoss(args['loss_computer'], args['temperature'], args).to(self.device)

        # Log Utils
        self.dump_folder = f'{self.args["dump_folder"]}-model'
        self.tb_writer = SummaryWriter(logdir=self.args['dump_folder'])


    def run_train_epoch(self, epoch_idx):
        for batch_id, batch_graph_idx in enumerate(self.data.train_dataloader()):
            if self.args['generator_process'] == 'before_encoder':
                graphs_i, graphs_j = [], []
                with self.env.begin(write=False) as txn:
                    for idx in batch_graph_idx:
                        graph = pickle.loads(txn.get(str(idx.item()).encode(), db=self.graph_db))
                        kgraph = pickle.loads(txn.get(str(idx.item()).encode(), db=self.kgraph_db))
                        graphs_i.append(graph)
                        graphs_j.append(kgraph)

                graph_batch_i = dgl.batch(graphs_i).to(self.device)
                graph_batch_j = dgl.batch(graphs_j).to(self.device)

                graph_embedding_i = self.projector(self.readout(graph_batch_i, self.encoder(graph_batch_i)))
                graph_embedding_j = self.projector(self.kreadout(graph_batch_j, self.kencoder(graph_batch_j)))

            loss = self.criterion(graph_embedding_i, graph_embedding_j)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            logger.info(f'[{epoch_idx:03d}/{batch_id:03d}] train loss {loss.item():.4f}')
            self.tb_writer.add_scalar('training_loss', loss.item(), self.global_step)

            # save model
            if self.global_step % 2000 == 0:
            # if self.global_step % 20 == 0:
                snapshot(self.encoder, self.global_step, self.dump_folder)
                snapshot(self.readout, self.global_step, self.dump_folder)
                snapshot(self.kencoder, self.global_step, self.dump_folder)
                snapshot(self.kreadout, self.global_step, self.dump_folder)
            if self.global_step % self.step_per_schedule == 0:
                self.scheduler.step()


    def run(self):
        for epoch_idx in range(self.epoch_num):
            self.run_train_epoch(epoch_idx)

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--data_name', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--featurizer_type', type=str, default='random')

    parser.add_argument('--encoder_name', type=str, default='GNN')
    parser.add_argument('--node_indim', type=int, default=128)
    parser.add_argument('--edge_indim', type=int, default=64)
    parser.add_argument('--hidden_feats', type=int, default=512)
    parser.add_argument('--node_hidden_feats', type=int, default=64)
    parser.add_argument('--edge_hidden_feats', type=int, default=128)
    parser.add_argument('--num_step_message_passing', type=int, default=6)
    parser.add_argument('--gnn_norm', type=str, default=None) # None, both, right
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--batchnorm', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_gnn_layers', type=int, default=4)

    parser.add_argument('--generator_process', type=str, default='before_encoder')
    parser.add_argument('--generator_level', type=str, default='graph')
    parser.add_argument('--loss_computer', type=str, default='nce_softmax')
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_per_schedule', type=int, default=500)

    parser.add_argument('--lmdb_env', type=str, default='../data/zinc15_250K_2D_random')
    parser.add_argument('--lmdb_db', type=str, default='graph')

    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument("--dump_path", default="dumped", type=str,
                        help="Experiment dump path")
    parser.add_argument("--exp_name", default="", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")

    parser.add_argument('--initial_path', type=str, default='initial/RotatE_128_64_emb.pkl')
    parser.add_argument('--encoder_path', type=str, default=None)
    parser.add_argument('--readout_path', type=str, default=None)
    parser.add_argument('--kencoder_path', type=str, default=None)
    parser.add_argument('--kreadout_path', type=str, default=None)

    return parser.parse_args().__dict__




if __name__ == '__main__':
    args = get_args()
    set_seed(args['seed'])

    logger, dump_folder = initialize_exp(Namespace(**args))
    args['dump_folder'] = dump_folder
 
    pretrainer = Pretrainer(args)
    pretrainer.run()