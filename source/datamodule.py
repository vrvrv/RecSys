from collections import defaultdict

import numpy as np
import pytorch_lightning as pl

class Loader:
    def __init__(self, adj, args):
        self.adj = adj
        self.args = args
        
class pl_Loader(pl.LightningDataModule):
    def __init__(adj, args):
        super().__init__()
        
        self.pos_user_msg_pair = np.asarray((adj>0).nonzero()).T
        
        usr, msg = (adj<0).nonzero()
        self.neg_usr_msg_pair = defaultdict(list)

        for u, m in zip(usr, msg):
            self.neg_usr_msg_pair[m].append(u)
            
        self.is_nsml = args.nsml
        
        if self.is_nsml :
            import nsml
            
            self.idx2usr_id = np.load(os.path.join(nsml.DATASET_PATH, 
                                                   "Jinhwan/git/talktalk-ctr/data/user_list.npy"))
            
        else :
            self.idx2usr_id = np.load("data/users/user_list.npy")

            
        self.n_worker = 4 * args.cpus
        self.batch_size = args.batch_size
        
            
        
class PairLoader(pl_Loader):
    def __init__(self, adj, args):
        super().__init__(adj, args)

    def setup(self, stage=None):
        
        self.train = PairDataset(self.pos_user_msg_pair, 
                                 self.neg_usr_msg_pair,
                                 self.idx2usr_id, 
                                 self.n_negative, 
                                 train_or_test = 'train',
                                 is_nsml = self.is_nsml)

    def train_dataloader(self):
        return DataLoader(self.train, 
                          num_workers = self.n_worker, 
                          batch_size=self.batch_size,
                          shuffle = True)
        
    
    
    
class TripletLoader(pl_Loader):
    def __init__(self, adj, args):
        super().__init__(adj, args)
        
        self.n_negative = args.n_negative

    def setup(self, stage=None):
        
        self.train = TripletDataset(self.pos_user_msg_pair, 
                                    self.neg_usr_msg_pair,
                                    self.idx2usr_id, 
                                    self.n_negative, 
                                    train_or_test = 'train',
                                    is_nsml = self.is_nsml)

    def train_dataloader(self):
        return DataLoader(self.train, 
                          num_workers = self.n_worker, 
                          batch_size=self.batch_size,
                          shuffle = True)
        
        
        
    
    