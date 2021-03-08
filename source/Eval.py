import numpy as np
from tqdm import tqdm

def precision_K(pred, true):
    '''
    pred : 유저 추천목록 (길이 K)
    true : 실제로 클릭한 유저 목록 
    '''
    
    tp = np.intersect1d(pred, true)
    
    return len(tp)/len(pred)
    
    
def AP(pred, true):
    '''
    pred : 유저 추천목록 (길이 K)
    true : 실제로 클릭한 유저 목록 
    '''
    ap = 0
    
    K = len(pred)
    m = 0 
    
    for k in range(1, K+1) :
        
        if pred[k-1] in true :
            m +=1
            ap += precision_K(pred[:k], true)
        
    if m == 0 :
        return 0
    
    return ap/m

def MAP(pred, true_adj):
    '''
    pred : 유저 추천목록 (K x I)
    true_adj : 실제 인접행렬 (U x I)
    '''
    U, I = true_adj.shape
    
    ap_sum = 0
    
    pbar = tqdm(range(I))
    
    for i in pbar:
        
        view_i, _ = (true_adj[:,i] != 0).nonzero()
        true_i, _ = (true_adj[:,i] == 1).nonzero()
        
        ap_i = AP(pred[i], true_i)
        
        ap_sum += ap_i
        
        pbar.set_description("MAP is %.8f" % (ap_sum/(i+1)))
        
    return ap_sum/I


def RR(pred, true):
    '''
    pred : 유저 추천목록 (길이 K)
    true : 실제로 클릭한 유저 목록 
    '''
    rr = 0
    
    K = len(pred)
    m = 0 
    
    for k in range(1, K+1) :
        
        if pred[k-1] in true :
            m +=1
            rr += 1/k
        
    if m == 0 :
        return 0
    
    return rr/m

def MRR(pred, true_adj):
    '''
    pred : 유저 추천목록 (K x I)
    true_adj : 실제 인접행렬 (U x I)
    '''
    U, I = true_adj.shape
    
    rr_sum = 0
    
    pbar = tqdm(range(I))
    
    for i in pbar:
        
        view_i, _ = (true_adj[:,i] != 0).nonzero()
        true_i, _ = (true_adj[:,i] == 1).nonzero()
        
        rr_i = RR(pred[i], true_i)
        
        rr_sum += rr_i
        
        pbar.set_description("MRR is %.8f" % (rr_sum/(i+1)))
        
    return rr_sum/I
    

def HR(pred, true_adj):
    
    _, K = pred.shape
    
    hit = 0
    
    U, I = true_adj.shape
    
    pbar = tqdm(range(I))
    
    for i in pbar:
        
        view_i, _ = (true_adj[:,i] != 0).nonzero()
        true_i, _ = (true_adj[:,i] == 1).nonzero()
        
        hit += len(np.intersect1d(pred[i], true_i))/K
            
        pbar.set_description("Hit Ratio is %.8f" % (hit/(i+1)))
    
    
    return hit/I


def COVERAGE(pred, true_adj):
    pass


def ENTROPY(pred, true_adj):
    pass


def AUROC(true_adj):
    pass



def evaluate_all(pred, true_adj):
    
    _map = MAP(pred, true_adj)
    _mrr = MRR(pred, true_adj)
    _hr = HR(pred, true_adj)
    
    return {'MAP' : _map, 'MRR' : _mrr, "HR" : _hr}

    