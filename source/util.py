from tqdm import tqdm

def rcplist(true_adj) -> list:
    
    non_zero_row, non_zero_col = true_adj.T.nonzero()
    
    n_user, n_msg = true_adj.shape
    
    rcp_list = [non_zero_col[non_zero_row==i] for i in tqdm(range(n_msg))]
    
    return rcp_list


def module_import(module_name):
    try:
        return __import__(module_name)
    except ImportError:
        return None 