import os
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset import *
from .util import get_current_path



class Loader:
    def __init__(self, adj, args):
        self.adj = adj
        self.args = args
        
class pl_Loader(pl.LightningDataModule):
    def __init__(self, adj, args):
        super().__init__()
        
        self.pos_user_msg_pair = np.asarray((adj>0).nonzero()).T
        
        usr, msg = (adj<0).nonzero()
        self.neg_usr_msg_pair = defaultdict(list)

        for u, m in zip(usr, msg):
            self.neg_usr_msg_pair[m].append(u)
            
        DATASET_PATH = get_current_path(args.path, args.nsml)
        self.idx2usr_id = np.load(os.path.join(DATASET_PATH, "data/users/user_list.npy"))

        self.n_worker = 4 * args.cpus
        self.batch_size = args.batch_size
        
            
        
class PairLoader(pl_Loader):
    def __init__(self, adj, args):
        super().__init__(adj, args)

    def setup(self, stage=None):
        
        self.train = PairDataset(self.pos_user_msg_pair, 
                                 self.neg_usr_msg_pair,
                                 self.idx2usr_id,
                                 args = self.args, 
                                 train_or_test = 'train')

    def train_dataloader(self):
        return DataLoader(self.train, 
                          num_workers = self.n_worker, 
                          batch_size=self.batch_size,
                          shuffle = True)
        
    
    
    
class TripletLoader(pl_Loader):
    def __init__(self, adj, args):
        super().__init__(adj, args)
        
        self.n_negative = args.n_negative
        self.args = args

    def setup(self, stage=None):
        
        self.train = TripletDataset(self.pos_user_msg_pair, 
                                    self.neg_usr_msg_pair,
                                    self.idx2usr_id, 
                                    args = self.args,
                                    train_or_test = 'train')

    def train_dataloader(self):
        return DataLoader(self.train, 
                          num_workers = self.n_worker, 
                          batch_size=self.batch_size,
                          shuffle = True)
        
        
        
    
    