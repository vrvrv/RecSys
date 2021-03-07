from argparse import ArgumentParser
    
def parser():
    
    parser = ArgumentParser(description="The parent parser")
    parser.add_argument('--nsml', action='store_true')
    parser.add_argument('--cpus', type = int, default=1)
    parser.add_argument('--cut-off', dest = 'K', type = int, default=1)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model', type=str, default = 'Random')
    parser.add_argument('--adj-path', dest = 'ADJ', type = str, default='./data/adjacency_matrix')
    
    return parser



def ctr_predict(parser):
    from source.trainer import fit_model
    
    model = fit_model(parser)
    pred = model.infer_top_K()
    
    if args.eval :
        evaluate(pred)
    


if __name__ == '__main__':
    
    _parser = parser()
    
    ctr_predict(
        parser = _parser
    )