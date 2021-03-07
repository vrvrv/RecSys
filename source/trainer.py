import os
from source import datamodule, model

#import dataloader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def subparser(parser):
    subparsers = parent_parser.add_subparsers(title="subparser")

def Model_Info(model):
    
    if model == 'Toppop': info = {'type' : 'rb', 'datamodule' : 'Loader'}
    elif model == 'Random': info = {'type' : 'rb', 'datamodule' : 'Loader'}
    elif model == 'Logistic' : info = {'type' : 'mb', 'datamodule' : 'PairLoader'}
    elif model == 'SymML' : info = {'type' : 'nn', 'datamodule' : 'TripletLoader'}
    else :
        raise NameError
        
    return info
    
    
def fit_model(parser):
    from scipy.sparse import load_npz

    model_name = parser.parse_args().model
    
    _model = getattr(model, model_name)
    
    parser = _model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )
    
    model_info = Model_Info(model_name)
    
    # Train dataset
    train_adj = load_npz(os.path.join(parser.parse_args().ADJ, 'train_adj.npz'))
    n_usr, n_msg = train_adj.shape
    
    # parse arguments
    args = parser.parse_args()
    
    
    # Define datamodule
    _datamodule = getattr(datamodule, model_info['datamodule'])

    train_module = _datamodule(train_adj, args)
    
    # Define model
    __model = _model(n_usr, n_msg, **vars(args))
    
    
    if model_info['type'] is 'nn':
    
        # Define Callback
        checkpoint_callback = ModelCheckpoint(save_top_k = -1)

        logger = TensorBoardLogger(
            version = f'{model_name}',
            name='lightning_logs'
        )

        trainer = Trainer.from_argparse_args(args,
                                             callbacks=[checkpoint_callback],
                                             logger = logger)

        trainer.fit(__model, train_module)
        
    else :
        __model.fit(train_module)
    
    return __model