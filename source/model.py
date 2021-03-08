import os
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm
from scipy.sparse import load_npz
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from source import embedding
from source.util import *

class pl_model(pl.LightningModule):
    def __init__(self, n_usr, n_msg, **kwargs):
        super().__init__()
        
        self.n_usr = n_usr
        self.n_msg = n_msg
        
        self.hparams = kwargs
        self.DATASET_PATH = get_current_path(self.hparams.path, self.hparams.nsml)
        
    def load_test_set(self):
        
        test_adj = load_npz(os.path.join(self.DATASET_PATH,
                                         self.hparams.ADJ, 
                                         'test_adj.npz'))
        
        test_msg_list = np.load(os.path.join(self.DATASET_PATH,
                                             self.hparams.ADJ, 
                                             'test_msg_list.npy'))
        
        return test_adj, test_msg_list
        
        
    def infer_top_K(self, K : int, evaluate : bool, test_set = None):
            
        if test_set is None :
            test_adj, test_msg_list = self.load_test_set()
           
        rcps = rcplist(test_adj)
            
        self.pred = self.inference(rcps, test_msg_list, K)
            
        if evaluate : 
            self.eval_metric(test_adj, ['MAP', 'MRR', 'HR'])
            
    def eval_metric(self, test_adj, metrics):
        
        from source import Eval
        
        result = dict()
        
        for metric in metrics :
            _qeval = getattr(Eval, metric)
            
            result[metric] = _qeval(self.pred, test_adj)
        
        print(result)
        
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser
    
    def inference(self, test_adj, test_msg_list, K) -> np.array:
        pass
    
        
    
class Toppop(pl_model):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--order_by', type=str, default='ctr')
        return parser   
    
    def fit(self, train_module):
        
        train_adj, args = train_module.adj, train_module.args
        
        # compute popularity
        
        if args.order_by == 'ctr' :
            self.ord = np.array((train_adj>0).sum(axis=1)/(np.abs(train_adj).sum(axis=1)+1e-3)).squeeze()
            
        elif args.order_by == 'n_click' :
            self.ord = np.array((train_adj>0).sum(axis=1)).squeeze()

    def inference(self, rcplist, test_msg_list, K):

        pred = np.empty((len(test_msg_list), K))

        for i in tqdm(range(len(test_msg_list))):
            pred[i] = rcplist[i][self.ord[rcplist[i]].argsort()[-K:][::-1]]      
            
        return pred
    
    
class Random(pl_model):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--replace', action='store_true')
        return parser   
    
    
    def fit(self, train_module):
        
        self.replace = train_module.args.replace

    
    def inference(self, rcplist, test_msg_list, K) -> np.array:

        pred = np.empty((len(test_msg_list), K))

        for i in tqdm(range(len(test_msg_list))):
            pred[i] = np.random.choice(rcplist[i], 
                                       size = K,
                                       replace = self.replace)
            
        return pred
        
        
class SVM(pl_model):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--usr_dim', type = int, default = 392)
        parser.add_argument('--msg_dim', type = int, default = 768)
        parser.add_argument('--optimizer', type = str, default = 'SGD')
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--learning_rate', type = float, default = 0.1)
        parser.add_argument('--weight_decay', type = float, default = 0.01)
        
        return parser   
    
    def __init__(self, n_usr, n_msg, **kwargs):
        super().__init__(n_usr, n_msg, **kwargs)
        
        self.Linear = nn.Linear(self.hparams.usr_dim + self.hparams.msg_dim, 1)
        
    def forward(self, input):
        x = self.Linear(input)
        return x
            
    def training_step(self, batch, batch_idx):
        usr, msg, y = batch
        output = self(torch.cat([usr, msg], axis=1))
        
        return torch.mean(torch.clamp(1 - output*(2*y-1), min = 0))
    
    
    def configure_optimizers(self):
        
        optim = getattr(torch.optim, self.hparams.optimizer)
        return optim(self.parameters(), 
                     lr = self.hparams.learning_rate, 
                     weight_decay = self.hparams.weight_decay)   
    
    def inference(self, rcplist, test_msg_list, K) -> np.array:

        pred = np.empty((len(test_msg_list), K))
        
        score = defalutdict(list)
        
        i = 0
        
        for rcp, msg_seq in tqdm(zip(rcplist, test_msg_list), total = len(test_msg_list)):
            
            DATAPATH = os.path.join(self.DATASET_PATH, f"data/items/{msg_seq}.npy")
            
            msg = torch.from_numpy(np.load(DATAPATH)).float()

            for nv_id in rcp :
                DATAPATH = os.path.join(self.DATASET_PATH,
                                        f"data/users/starts_with_{nv_id[:2]}/{nv_id}.npy")
                
                usr = torch.from_numpy(np.load(DATAPATH)).float()
                score[msg_seq].append(self(torch.cat([usr, msg], axis=1)))
                topk = rcp[torch.topk(score[msg_seq], K).indices]

            pred[i] = np.array(topk)
            
            i += 1
        
        return pred
    

def ContrastiveLoss(pos_dist, neg_dist, margin):
    
    return torch.nn.ReLU()(pos_dist.unsqueeze(1)-neg_dist+margin).mean()


class CML(pl_model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type = int, default = 24)
        parser.add_argument('--usr_dim', type = int, default = 392)
        parser.add_argument('--msg_dim', type = int, default = 768)
        parser.add_argument('--n_negative', type = int, default = 8)
        parser.add_argument('--margin', type = float, default = 0.99)
        parser.add_argument('--dropout_prob', type = float, default = 0.2)
        parser.add_argument('--lambda_f', type = float, default = 0.5)
        parser.add_argument('--lambda_c', type = float, default = 0.1)
        parser.add_argument('--optimizer', type = str, default = 'AdamW')
        parser.add_argument('--learning_rate', type = float, default = 1e-4)
        parser.add_argument('--weight_decay', type = float, default = 1e-2)
        parser.add_argument('--batch_size', type = int, default = 32)
        parser.add_argument('--usr_proj', type = str, default = 'mlp_embedding_usr')
        parser.add_argument('--msg_proj', type = str, default = 'mlp_embedding_msg')
        
        return parser
        
        
    def __init__(self, n_usr, n_msg, **kwargs):
        super().__init__(n_usr, n_msg, **kwargs)
        
        self.embedding1 = nn.Embedding(n_usr, self.hparams.hidden_dim)
        self.embedding2 = nn.Embedding(n_msg, self.hparams.hidden_dim)
        
        self.usr_proj = getattr(embedding, self.hparams.usr_proj)(self.hparams)
        self.msg_proj = getattr(embedding, self.hparams.msg_proj)(self.hparams)
        

    def training_step(self, batch, batch_idx):
        pos_pairs, (neg_usr_idx, neg_usr) = batch
        
        (pos_usr_idx, pos_usr), (pos_msg_idx, pos_msg) = pos_pairs
        
        pos_dist = self._distance(pos_usr_idx, pos_msg_idx, "pos")
        neg_dist = self._distance(neg_usr_idx, pos_msg_idx, "neg")
        
        msg_centric_loss = ContrastiveLoss(pos_dist, 
                                           neg_dist, 
                                           self.hparams.margin)

        loss = msg_centric_loss \
             + self.hparams.lambda_f * self.reg1(pos_pairs, neg_usr_idx, neg_usr) \
             + self.hparams.lambda_c * self.reg2(pos_usr_idx, pos_msg_idx, neg_usr_idx)
        
        self.log('train_loss', loss)
        
        return loss
    
    def _distance(self, idx1, idx2, pos_or_neg, usr_usr = False):
        
        
        if usr_usr : 
            embed1 = self.embedding1(idx1)
            embed2 = self.embedding1(idx2)
            
        else :
            embed1 = self.embedding1(idx1)
            embed2 = self.embedding2(idx2)
        
        
        if pos_or_neg == "pos" :
            
            return torch.square(embed1-embed2).mean(axis=1)
            
        else :
            return torch.square(embed1-embed2.unsqueeze(1)).mean(axis=-1)
    
    
    def reg1(self, pos_pairs, neg_usr_idx, neg_usr):
        
        (pos_usr_idx, pos_usr), (pos_msg_idx, pos_msg) = pos_pairs
        
        regularization = torch.square(self.embedding1(pos_usr_idx) - self.usr_proj(pos_usr)).mean()
        
        regularization += torch.square(self.embedding1(neg_usr_idx) - self.usr_proj(neg_usr)).mean()        
        
        regularization += torch.square(self.embedding2(pos_msg_idx) - self.msg_proj(pos_msg)).mean()        
        
        return regularization
        
    def reg2(self, pos_usr_idx, pos_msg_idx, neg_usr_idx):
        usr_idx = torch.cat([pos_usr_idx, torch.unique(neg_usr_idx.flatten())])
        msg_idx = pos_msg_idx
        
        y = torch.cat([self.embedding1(usr_idx), self.embedding2(msg_idx)], axis=0)
        
        C = (y - y.mean(axis=0)).T.matmul(y - y.mean(axis=0))/y.size(0)
        
        return (torch.sqrt(torch.square(C).sum()) - torch.square(torch.diag(C)).sum())/y.size(0)
        

    def configure_optimizers(self):
        
        optim = getattr(torch.optim, self.hparams.optimizer)
        return optim(self.parameters(), 
                     lr = self.hparams.learning_rate, 
                     weight_decay = self.hparams.weight_decay)

    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        
        optimizer.step(closure=optimizer_closure)
        
        with torch.no_grad():
            
            self.embedding1.weight.data /= torch.clamp(torch.sqrt((self.embedding1.weight.data**2).sum(axis=1,
                                                                                                       keepdim=True)),
                                                min=1)
            
            self.embedding2.weight.data /= torch.clamp(torch.sqrt((self.embedding2.weight.data**2).sum(axis=1,
                                                                                                       keepdim=True)),
                                                min=1)
            
    def inference(self, rcplist, test_msg_list, K) -> np.array:
            
        pred = torch.empty((len(test_msg_list), K))

        for i, msg_seq in enumerate(tqdm(test_msg_list)):
            msg_emb = np.load(os.path.join(self.DATASET_PATH, f"data/items/{msg_seq}.npy"))
            msg_emb = self.msg_proj(torch.from_numpy(msg_emb).float()).data

            dist = self.embedding1(torch.tensor(rcplist[i], dtype = torch.long)) - msg_emb
            
            topk_idx = torch.topk(-torch.square(dist).sum(axis=1), K).indices.tolist()
            topk = [rcplist[i][idx] for idx in topk_idx]
            pred[i] = np.array(topk)

        return pred
    
class SymML(CML):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = CML.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument('--lambda_0', type = float, default = 0.5)
        parser.add_argument('--lambda_1', type = float, default = 0.3)
        parser.add_argument('--lambda_2', type = float, default = 0.1)
        parser.add_argument('--l', type = float, default = 1)
        
        return parser
        
        
    def __init__(self, n_usr, n_msg, **kwargs):
        super().__init__(n_usr, n_msg, **kwargs)
        
        self.usr_margin = nn.Parameter(torch.zeros(n_usr))
        self.msg_margin = nn.Parameter(torch.zeros(n_msg))
        

    def training_step(self, batch, batch_idx):
        pos_pairs, (neg_usr_idx, neg_usr) = batch
        
        (pos_usr_idx, pos_usr), (pos_msg_idx, pos_msg) = pos_pairs
        
        pos_dist = self._distance(pos_usr_idx, pos_msg_idx, "pos")
        neg_dist = self._distance(neg_usr_idx, pos_msg_idx, "neg")
        msg_dist = self._distance(neg_usr_idx, pos_usr_idx, "neg", usr_usr = True)
        
        msg_centric_loss = ContrastiveLoss(pos_dist, 
                                           neg_dist, 
                                           torch.exp(self.msg_margin[pos_msg_idx]).unsqueeze(-1))

        usr_centric_loss = ContrastiveLoss(pos_dist, 
                                           msg_dist, 
                                           torch.exp(self.usr_margin[neg_usr_idx]))
        
        loss = msg_centric_loss \
             + self.hparams.lambda_0 * usr_centric_loss \
             + self.hparams.lambda_1 * self.reg1(pos_pairs, neg_usr_idx, neg_usr) \
             + self.hparams.lambda_2 * self.reg2()
        
        self.log('train_loss', loss)
        
        return loss
    
        
    def reg2(self):
        return -(torch.exp(self.usr_margin).mean() + torch.exp(self.msg_margin).mean())
    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        
        optimizer.step(closure=optimizer_closure)
        
        with torch.no_grad():
            self.usr_margin.clamp_(max = self.hparams.l)
            self.msg_margin.clamp_(max = self.hparams.l)
            
            self.embedding1.weight.data /= torch.clamp(torch.sqrt((self.embedding1.weight.data**2).sum(axis=1,
                                                                                                       keepdim=True)),
                                                min=1)
            
            self.embedding2.weight.data /= torch.clamp(torch.sqrt((self.embedding2.weight.data**2).sum(axis=1,
                                                                                                       keepdim=True)),
                                                min=1)
            
    