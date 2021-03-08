import torch.nn as nn

def mlp_embedding_usr(hparams):
    return nn.Sequential(
        nn.Linear(hparams.usr_dim, hparams.hidden_dim),
        nn.Tanh()
    )

def mlp_embedding_msg(hparams):
    return nn.Sequential(
        nn.Linear(hparams.msg_dim, 128),
        nn.Tanh(),
        nn.Dropout(p=hparams.dropout_prob),
        nn.Linear(128, hparams.hidden_dim),
        nn.Tanh()
    )