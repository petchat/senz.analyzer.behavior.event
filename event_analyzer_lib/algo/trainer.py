from hmmlearn.hmm import GMMHMM
from datasets import Dataset
from sklearn.mixture.gmm import GMM
import numpy as np

def trainingGMMHMM(
        dataset,               # training dataset.
        n_c,                   # number of hmm's components (ie. hidden states)
        n_m,                   # number of gmm's mixtures (ie. Gaussian model)
        start_prob_prior=None, # prior of start hidden states probabilities.
        trans_mat_prior=None,  # prior of transition matrix.
        start_prob=None,       # the start hidden states probabilities.
        trans_mat=None,        # the transition matrix.
        gmms=None,             # models' params of gmm
        covar_type='full',
        n_i=50
):
    # Initiation of dataset.
    # d = Dataset(dataset)
    X = dataset.getDataset()
    # Initiation of GMM.
    _GMMs = []
    if gmms is None:
        _GMMs = None
    else:
        for gmm in gmms:
            _GMM = GMM(n_components=n_m, covariance_type=covar_type)
            _GMM.covars_  = np.array(gmm["covars"])
            _GMM.means_   = np.array(gmm["means"])
            _GMM.weights_ = np.array(gmm["weights"])
            _GMMs.append(_GMM)
    # Initiation of GMMHMM.
    model = GMMHMM(
        startprob_prior=np.array(start_prob_prior),
        transmat_prior=np.array(trans_mat_prior),
        startprob=np.array(start_prob),
        transmat=np.array(trans_mat),
        gmms=_GMMs,
        n_components=n_c,
        n_mix=n_m,
        covariance_type=covar_type,
        n_iter=n_i
    )
    # Training.
    model.fit(X)
    # The result.
    new_gmmhmm = {
        "nComponent": n_c,
        "nMix": n_m,
        "covarianceType": covar_type,
        "hmmParams": {
            "startProb": model.startprob_.tolist(),
            "transMat": model.transmat_.tolist()
        },
        "gmmParams": {
            "nMix": n_m,
            "covarianceType": covar_type,
            "params": []
        }
    }

    for i in range(0, n_m):
        gaussian_model = {
            "covars": model.gmms_[i].covars_.tolist(),
            "means":  model.gmms_[i].means_.tolist(),
            "weights": model.gmms_[i].weights_.tolist()
        }
        new_gmmhmm["gmmParams"]["params"].append(gaussian_model)

    return new_gmmhmm

if __name__ == '__main__':
    pass