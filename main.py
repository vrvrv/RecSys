import os
from argparse import ArgumentParser
from source import model
    
    
def parser(model):
    from pytorch_lightning import Trainer
    
    parser = ArgumentParser(description="The parent parser")
    parser.add_argument('--nsml', action='store_true')
    parser.add_argument('--cpus', type = int, default=1)
    parser.add_argument('--cut-off', dest = 'K', type = int, default=1)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--path', type = str, default='Jinhwan/git/RecSys')
    parser.add_argument('--adj-path', dest = 'ADJ', type = str, default='./data/adjacency_matrix')
    
    parser = model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(
        parser.add_argument_group(title="pl.Trainer args")
    )
    
    return parser.parse_args()


def main(model, args):
    from source.trainer import fit_model
    
    fitted_model = fit_model(model, args)
    pred = fitted_model.infer_top_K()
    
    if args.eval : evaluate(pred)
    

if __name__ == '__main__':
    
    _model = getattr(model, 'SymML')
    _args = parser(_model)
    
    main(_model, _args)
    
    