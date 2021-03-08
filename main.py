import os
import logging
from argparse import ArgumentParser
from source import model
    
    
def parser(model):
    from pytorch_lightning import Trainer
    
    parser = ArgumentParser(description="The parent parser")
    parser.add_argument('--nsml', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cpus', type = int, default=1)
    parser.add_argument('--cut-off', dest = 'K', type = int, default=30)
    parser.add_argument('--path', type = str, default=os.getcwd())
    parser.add_argument('--train_or_test', type = str, default='train')
    parser.add_argument('--adj-path', dest = 'ADJ', type = str, default='data/adjacency_matrix')
    
    parser = model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )
    
    return parser.parse_args()


def main(model, args):
    from source.trainer import fit_model
    
    print(f"cpus : {args.cpus}, gpus : {args.gpus if type(args.gpus)==int else 0}")
    if args.eval : print(f"cut off : {args.K}") 
    
    fitted_model = fit_model(model, args)
    fitted_model.infer_top_K(K = args.K, evaluate = args.eval)
    

if __name__ == '__main__':
    
    _model = getattr(model, 'SymML')
    _args = parser(_model)
    
    main(_model, _args)
    
    