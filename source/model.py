import os
from argparse import ArgumentParser

from tqdm import tqdm
from scipy.sparse import load_npz
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from source import embedding
from .util import rcplist


class pl_model(pl.LightningModule):
    def __init__(self, n_usr, n_msg, **kwargs):
        super().__init__()
        
        self.n_usr = n_usr
        self.n_msg = n_msg
        
        self.hparams = kwargs
        
    def load_test_set(self):
        test_adj = load_npz(os.path.join(self.hparams.ADJ, 'test_adj.npz'))
        test_msg_list = np.load(os.path.join(self.hparams.ADJ, 'test_msg_list.npz'))
        
        return test_adj, test_msg_list
        
    def infer_top_K(self, K):
            
        if test_set is None :
            test_adj, test_msg_list = self.load_test_set()

        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser
        
    
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

        
    def infer_top_K(self, K, test_set = None):
            
        if test_set is None :
            test_adj, test_msg_list = self.load_test_set()
            
        rcp_list = rcplist(test_adj)
        n_msg = len(test_msg_list)

        pred = np.empty((n_msg, K))

        for i in tqdm(range(n_msg)):
            pred[i] = rcp_list[i][self.ord[rcp_list[i]].argsort()[-K:][::-1]]      
            
        return pred
    
    
class Random(pl_model):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--replace', action='store_true')
        return parser   
    
    
    def fit(self, train_module):
        
        self.replace = train_module.args.replace

    
    def infer_top_K(self, K, test_set = None):
            
        if test_set is None :
            test_adj, test_msg_list = self.load_test_set()
        
        rcp_list = rcplist(test_adj)
        
        n_test_msg = len(test_msg_list)

        pred = np.empty((n_msg, K))

        for i in tqdm(range(n_test_msg)):
            pred[i] = np.random.choice(rcp_list[i], 
                                       size = K,
                                       replace = self.replace)
            
        return pred
        
        
    
class SymML(pl_model):

    @staticmethod
    def add_model_specific_args(parent_parser):
        print("hello")
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type = int, default = 12)
        parser.add_argument('--usr_dim', type = int, default = 392)
        parser.add_argument('--msg_dim', type = int, default = 12)
        parser.add_argument('--n_negative', type = int, default = 1)
        parser.add_argument('--dropout_prob', type = float, default = 0.2)
        parser.add_argument('--lambda_e', type = float, default = 0.5)
        parser.add_argument('--lambda_f', type = float, default = 0.3)
        parser.add_argument('--lambda_g', type = float, default = 0.1)
        parser.add_argument('--l', type = float, default = 1)
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
        
        self.usr_margin = nn.Parameter(torch.zeros(n_usr))
        self.msg_margin = nn.Parameter(torch.zeros(n_msg))
        
        self.usr_proj = getattr(embedding, self.hparams.usr_proj)(self.hparams)
        self.msg_proj = getattr(embedding, self.hparams.msg_proj)(self.hparams)

        
        
    def forward(self, msg_feat, reciptant, cutoff):
        msg_embedding = self.embedding(msg_feat)
        
        candidates = self.embedding1(torch.tensor(reciptant, dtype = torch.long))
        
        nearest_naver = abs(candidates-msg_embedding).topk(cutoff)
        
        return nearest_naver
        

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
             + self.hparams.lambda_e * usr_centric_loss \
             + self.hparams.lambda_f * self.reg1(pos_pairs, neg_usr_idx, neg_usr) \
             + self.hparams.lambda_g * self.reg2()
        
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
        
        regularization = torch.square(self.embedding1(pos_usr_idx) - self.user_proj(pos_usr)).mean()
        
        regularization += torch.square(self.embedding1(neg_usr_idx) - self.user_proj(neg_usr)).mean()        
        
        regularization += torch.square(self.embedding2(pos_msg_idx) - self.msg_proj(pos_msg)).mean()        
        
        return regularization
        
    def reg2(self):
        return -(torch.exp(self.usr_margin).mean() + torch.exp(self.msg_margin).mean())

    
    def configure_optimizers(self):
        
        optim = getattr(torch.optim, self.hparams.optimizer)
        return optim(self.parameters(), 
                     lr = self.hparams.learning_rate, 
                     weight_decay = self.hparams.weight_decay)

    
    
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
            
            
    def inference_top_K(self, K, test_set = None) -> torch.Tensor:
        
        if test_set is None :
            test_adj, test_msg_list = self.load_test_set()
            
        recommend_list = torch.empty((len(test_msg_list), K))

        for i in tqdm(range(len(test_msg_list))):
            recp_user_list, _ = test_adj[:,i].nonzero()

            msg_seq = test_msg_list[i]
            msg_emb = np.load(f"data/txt_embed/{msg_seq}.npy")

            msg_emb = self.msg_proj(torch.from_numpy(msg_emb).float()).data

            dist = self.embedding1(torch.tensor(recp_user_list, dtype = torch.long)) - msg_emb

            topk = recp_user_list[torch.topk(-torch.square(dist).sum(axis=1), K).indices]

            recommend_list[i] = torch.from_numpy(topk)

        return recommend_list
    