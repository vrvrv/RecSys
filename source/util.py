import os
from tqdm import tqdm

def rcplist(true_adj) -> list:
    
    non_zero_row, non_zero_col = true_adj.T.nonzero()
    
    n_user, n_msg = true_adj.shape
    
    rcp_list = [non_zero_col[non_zero_row==i] for i in tqdm(range(n_msg))]
    
    return rcp_list


def get_current_path(path, is_nsml = False):
    try:
        nsml = __import__('nsml')
        if is_nsml :
            return os.path.join(nsml.DATASET_PATH, 
                                path)
        else :
            return path
    
    except ImportError:
        return path