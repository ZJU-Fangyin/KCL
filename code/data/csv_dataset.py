import dgl.backend as F
import numpy as np
import os
import torch

from dgl.data.utils import save_graphs, load_graphs

from dgllife.utils.io import pmap
import pdb

__all__ = ['MoleculeCSVDataset']

class MoleculeCSVDataset(object):

    def __init__(self, df, smiles_to_graph, smiles_column,
                 cache_file_path, task_names=None, load=False, log_every=1000, init_mask=True,
                 n_jobs=1):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(smiles_to_graph, 
                          load, log_every, init_mask, n_jobs)

        # Only useful for binary classification tasks
        self._task_pos_weights = None

    def _pre_process(self, smiles_to_graph, 
                     load, log_every, init_mask, n_jobs=1):

        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing dgl graphs from scratch...')
            if n_jobs > 1:
                self.graphs = pmap(smiles_to_graph,
                                   self.smiles,
                                   n_jobs=n_jobs)
            else:
                self.graphs = []
                for i, s in enumerate(self.smiles):
                    if (i + 1) % log_every == 0:
                        print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))
                    self.graphs.append(smiles_to_graph(s))

            # Keep only valid molecules
            self.valid_ids = []
            graphs = []
            for i, g in enumerate(self.graphs):
                if g is not None:
                    self.valid_ids.append(i)
                    graphs.append(g)
            self.graphs = graphs
            _label_values = self.df[self.task_names].values
            # np.nan_to_num will also turn inf into a very large number
            self.labels = F.zerocopy_from_numpy(
                np.nan_to_num(_label_values).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy(
                    (~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'mask': self.mask,
                                    'valid_ids': valid_ids})
            else:
                self.mask = None
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'valid_ids': valid_ids})

        self.smiles = [self.smiles[i] for i in self.valid_ids]

    def __getitem__(self, item):

        if self.mask is not None:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)

    def task_pos_weights(self, indices):
        """Get weights for positive samples on each task

        This should only be used when all tasks are binary classification.

        Returns
        -------
        Tensor of dtype float32 and shape (T)
            Weight of positive samples on all tasks
        """
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = F.sum(self.labels[indices], dim=0)
        num_indices = F.sum(self.mask[indices], dim=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]

        return task_pos_weights
