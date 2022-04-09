import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pdb

class LmdbDataset(Dataset):
    def __init__(self, data_dir, data_name) -> None:
        super(LmdbDataset, self).__init__()
        with open(f'{data_dir}/{data_name}.pkl', 'rb') as f:
            self.data_list = pickle.load(f)
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

class LmdbDataModule():
    def __init__(self, data_dir, data_name, batch_size: int) -> None:
        super().__init__()
        self.dataset = LmdbDataset(data_dir, data_name)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)