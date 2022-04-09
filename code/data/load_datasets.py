import pandas as pd

from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive

from .csv_dataset import MoleculeCSVDataset
from .smiles_to_dglgraph import smiles_2_dgl, smiles_2_kgdgl
import pdb 

class BBBP(MoleculeCSVDataset):

    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./bbbp_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/bbbp.zip'
        data_path = get_download_dir() + '/bbbp.zip'
        dir_path = get_download_dir() + '/bbbp'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/BBBP.csv')

        super(BBBP, self).__init__(df=df,
                                   smiles_to_graph=smiles_to_graph,
                                   smiles_column='smiles',
                                   cache_file_path=cache_file_path,
                                   task_names=['p_np'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=True,
                                   n_jobs=n_jobs)

        self.load_full = False
        self.names = df['name'].tolist()
        self.names = [self.names[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.names[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class BACE(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./bace_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/bace.zip'
        data_path = get_download_dir() + '/bace.zip'
        dir_path = get_download_dir() + '/bace'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/bace.csv')

        super(BACE, self).__init__(df=df,
                                   smiles_to_graph=smiles_to_graph,
                                   smiles_column='mol',
                                   cache_file_path=cache_file_path,
                                   task_names=['Class'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=True,
                                   n_jobs=n_jobs)

        self.load_full = False
        self.ids = df['CID'].tolist()
        self.ids = [self.ids[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class MUV(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./muv_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/muv.zip'
        data_path = get_download_dir() + '/muv.zip'
        dir_path = get_download_dir() + '/muv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/muv.csv')

        self.ids = df['mol_id'].tolist()
        self.load_full = False

        df = df.drop(columns=['mol_id'])

        super(MUV, self).__init__(df=df,
                                  smiles_to_graph=smiles_to_graph,
                                  smiles_column='smiles',
                                  cache_file_path=cache_file_path,
                                  load=load,
                                  log_every=log_every,
                                  init_mask=True,
                                  n_jobs=n_jobs)

        self.ids = [self.ids[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class ClinTox(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./clintox_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/clintox.zip'
        data_path = get_download_dir() + '/clintox.zip'
        dir_path = get_download_dir() + '/clintox'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/clintox.csv')

        super(ClinTox, self).__init__(df=df,
                                      smiles_to_graph=smiles_to_graph,
                                      smiles_column='smiles',
                                      cache_file_path=cache_file_path,
                                      load=load,
                                      log_every=log_every,
                                      init_mask=True,
                                      n_jobs=n_jobs)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class SIDER(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./sider_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/sider.zip'
        data_path = get_download_dir() + '/sider.zip'
        dir_path = get_download_dir() + '/sider'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/sider.csv')

        super(SIDER, self).__init__(df=df,
                                    smiles_to_graph=smiles_to_graph,
                                    smiles_column='smiles',
                                    cache_file_path=cache_file_path,
                                    load=load,
                                    log_every=log_every,
                                    init_mask=True,
                                    n_jobs=n_jobs)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class ToxCast(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./toxcast_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/toxcast.zip'
        data_path = get_download_dir() + '/toxcast.zip'
        dir_path = get_download_dir() + '/toxcast'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/toxcast_data.csv')

        super(ToxCast, self).__init__(df=df,
                                      smiles_to_graph=smiles_to_graph,
                                      smiles_column='smiles',
                                      cache_file_path=cache_file_path,
                                      load=load,
                                      log_every=log_every,
                                      init_mask=True,
                                      n_jobs=n_jobs)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class HIV(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./hiv_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/hiv.zip'
        data_path = get_download_dir() + '/hiv.zip'
        dir_path = get_download_dir() + '/hiv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/HIV.csv')

        self.activity = df['activity'].tolist()
        self.load_full = False

        df = df.drop(columns=['activity'])

        super(HIV, self).__init__(df=df,
                                  smiles_to_graph=smiles_to_graph,
                                  smiles_column='smiles',
                                  cache_file_path=cache_file_path,
                                  load=load,
                                  log_every=log_every,
                                  init_mask=True,
                                  n_jobs=n_jobs)

        self.activity = [self.activity[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.activity[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class Tox21(MoleculeCSVDataset):
    def __init__(self, smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./tox21_dglgraph.bin',
                 n_jobs=1):
        self._url = 'dataset/tox21.csv.gz'
        data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        df = pd.read_csv(data_path)
        self.id = df['mol_id']

        df = df.drop(columns=['mol_id'])

        self.load_full = False

        super(Tox21, self).__init__(df, smiles_to_graph, 
                                    smiles_column='smiles',
                                    cache_file_path=cache_file_path,
                                    load=load, log_every=log_every, n_jobs=n_jobs)

        self.id = [self.id[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.mask[item], self.id[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

class ESOL(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./esol_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/ESOL.zip'
        data_path = get_download_dir() + '/ESOL.zip'
        dir_path = get_download_dir() + '/ESOL'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/delaney-processed.csv')

        super(ESOL, self).__init__(df=df,
                                   smiles_to_graph=smiles_to_graph,
                                   smiles_column='smiles',
                                   cache_file_path=cache_file_path,
                                   task_names=['measured log solubility in mols per litre'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=False,
                                   n_jobs=n_jobs)

        self.load_full = False
        # Compound names in PubChem
        self.compound_names = df['Compound ID'].tolist()
        self.compound_names = [self.compound_names[i] for i in self.valid_ids]
        # Estimated solubility
        self.estimated_solubility = df['ESOL predicted log solubility in mols per litre'].tolist()
        self.estimated_solubility = [self.estimated_solubility[i] for i in self.valid_ids]
        # Minimum atom degree
        self.min_degree = df['Minimum Degree'].tolist()
        self.min_degree = [self.min_degree[i] for i in self.valid_ids]
        # Molecular weight
        self.mol_weight = df['Molecular Weight'].tolist()
        self.mol_weight = [self.mol_weight[i] for i in self.valid_ids]
        # Number of H-Bond Donors
        self.num_h_bond_donors = df['Number of H-Bond Donors'].tolist()
        self.num_h_bond_donors = [self.num_h_bond_donors[i] for i in self.valid_ids]
        # Number of rings
        self.num_rings = df['Number of Rings'].tolist()
        self.num_rings = [self.num_rings[i] for i in self.valid_ids]
        # Number of rotatable bonds
        self.num_rotatable_bonds = df['Number of Rotatable Bonds'].tolist()
        self.num_rotatable_bonds = [self.num_rotatable_bonds[i] for i in self.valid_ids]
        # Polar Surface Area
        self.polar_surface_area = df['Polar Surface Area'].tolist()
        self.polar_surface_area = [self.polar_surface_area[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.compound_names[item], self.estimated_solubility[item], \
                   self.min_degree[item], self.mol_weight[item], \
                   self.num_h_bond_donors[item], self.num_rings[item], \
                   self.num_rotatable_bonds[item], self.polar_surface_area[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

class FreeSolv(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./freesolv_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/FreeSolv.zip'
        data_path = get_download_dir() + '/FreeSolv.zip'
        dir_path = get_download_dir() + '/FreeSolv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/SAMPL.csv')

        super(FreeSolv, self).__init__(df=df,
                                       smiles_to_graph=smiles_to_graph,
                                       smiles_column='smiles',
                                       cache_file_path=cache_file_path,
                                       task_names=['expt'],
                                       load=load,
                                       log_every=log_every,
                                       init_mask=False,
                                       n_jobs=n_jobs)

        self.load_full = False

        # Iupac names
        self.iupac_names = df['iupac'].tolist()
        self.iupac_names = [self.iupac_names[i] for i in self.valid_ids]
        # Calculated hydration free energy
        self.calc_energy = df['calc'].tolist()
        self.calc_energy = [self.calc_energy[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                   self.iupac_names[item], self.calc_energy[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

class Lipophilicity(MoleculeCSVDataset):
    def __init__(self,
                 smiles_to_graph=smiles_2_dgl,
                 load=False,
                 log_every=1000,
                 cache_file_path='./lipophilicity_dglgraph.bin',
                 n_jobs=1):

        self._url = 'dataset/lipophilicity.zip'
        data_path = get_download_dir() + '/lipophilicity.zip'
        dir_path = get_download_dir() + '/lipophilicity'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/Lipophilicity.csv')

        super(Lipophilicity, self).__init__(df=df,
                                            smiles_to_graph=smiles_to_graph,
                                            smiles_column='smiles',
                                            cache_file_path=cache_file_path,
                                            task_names=['exp'],
                                            load=load,
                                            log_every=log_every,
                                            init_mask=False,
                                            n_jobs=n_jobs)

        self.load_full = False

        # ChEMBL ids
        self.chembl_ids = df['CMPD_CHEMBLID'].tolist()
        self.chembl_ids = [self.chembl_ids[i] for i in self.valid_ids]

    def __getitem__(self, item):

        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], self.chembl_ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]


if __name__ == '__main__':
    dataset = FreeSolv(smiles_2_dgl)
