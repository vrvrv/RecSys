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
            
        self.DATASET_PATH = get_current_path(args.path, args.nsml)

        self.idx2usr_id = np.load(os.path.join(self.DATASET_PATH, 
                                               "data/users/user_list.npy"))
        
        self.idx2msg_id = np.load(os.path.join(self.DATASET_PATH, 
                                               "data/adjacency_matrix",
                                               f"{args.train_or_test}_msg_list.npy"))

        self.train_or_test = args.train_or_test
        self.n_worker = 4 * args.cpus
        self.batch_size = args.batch_size
        
            
        
class PairLoader(pl_Loader):
    def __init__(self, adj, args):
        super().__init__(adj, args)

    def setup(self, stage=None):
        
        self.dataset = PairDataset(self.pos_user_msg_pair, 
                                  self.neg_usr_msg_pair,
                                  self.idx2usr_id,
                                  idx2msg_id = self.idx2msg_id,
                                  DATASET_PATH = self.DATASET_PATH)

    def train_dataloader(self):
        
        if self.train_or_test == 'train' :
            return DataLoader(self.dataset, 
                              num_workers = self.n_worker, 
                              batch_size=self.batch_size,
                              shuffle = True)
        
    def test_dataloader(self):
        
        if self.train_or_test == 'test' :
            return DataLoader(self.dataset, 
                              num_workers = self.n_worker, 
                              batch_size=self.batch_size,
                              shuffle = False)
            
    
    
class TripletLoader(pl_Loader):
    def __init__(self, adj, args):
        super().__init__(adj, args)
        
        self.n_negative = args.n_negative

    def setup(self, stage=None):
        
        self.dataset = TripletDataset(self.pos_user_msg_pair, 
                                      self.neg_usr_msg_pair,
                                      self.idx2usr_id, 
                                      idx2msg_id = self.idx2msg_id,
                                      DATASET_PATH = self.DATASET_PATH,
                                      n_negative = self.n_negative)

    def train_dataloader(self):
        
        if self.train_or_test == 'train' :
            return DataLoader(self.dataset, 
                              num_workers = self.n_worker, 
                              batch_size=self.batch_size,
                              shuffle = True)
        
    def test_dataloader(self):
        
        if self.train_or_test == 'test' :
            return DataLoader(self.dataset, 
                              num_workers = self.n_worker, 
                              batch_size=self.batch_size,
                              shuffle = True)
            
        
        
    
    