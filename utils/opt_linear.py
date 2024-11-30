import numpy as np

class LinearModel:
    def __init__(self, n_dense: int, n_sparse: int, p_feat: float) -> None:
        self.n_dense = n_dense
        self.n_sparse = n_sparse
        self.p_feat = p_feat
        # self.bias = False
        
    def ratio(self):
        return self.n_dense/self.n_sparse
    def p_feat(self):
        return self.p_feat
    def final_loss(self):
        theory_eigs = np.ones(self.n_sparse)*self.p_feat/3.0
        ans = np.sum(theory_eigs[int(self.n_dense)+1:])/self.n_sparse
        return ans
    

def multiple_models_losses(n_sparse, ratios, p_feats):
    models = []
    for ratio in ratios:
        for p_feat in p_feats:
            n_dense = max(1, int(ratio*n_sparse))
            models.append(LinearModel(n_dense, n_sparse, p_feat))
    return models