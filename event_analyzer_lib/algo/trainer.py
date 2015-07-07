from hmmlearn.hmm import GMMHMM
from sklearn.mixture import GMM
from datasets import Dataset
import numpy as np


def trainingGMMHMM(
        dataset,  # training dataset.
        n_c,  # number of hmm's components (ie. hidden states)
        n_m,  # number of gmm's mixtures (ie. Gaussian model)
        start_prob_prior=None,  # prior of start hidden states probabilities.
        trans_mat_prior=None,  # prior of transition matrix.
        start_prob=None,  # the start hidden states probabilities.
        trans_mat=None,  # the transition matrix.
        gmms=None,  # models' params of gmm
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
            _GMM.covars_ = np.array(gmm["covars"])
            _GMM.means_ = np.array(gmm["means"])
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
            "means": model.gmms_[i].means_.tolist(),
            "weights": model.gmms_[i].weights_.tolist()
        }
        new_gmmhmm["gmmParams"]["params"].append(gaussian_model)

    return new_gmmhmm


class GMMHMMTrainer(object):
    '''A wrapper to GMMHMM

    Attributes
    ----------
    _model: init params
    gmmhmm: hmmlearn GMMHMM instance
    params_: params after fit
    train_data_: current train datas
    '''

    def __init__(self, _model):
        hmm_params = _model['hmmParams']
        gmm_params = _model['gmmParams']
        n_iter = _model.get('nIter', 50)

        transmat = hmm_params['transMat']
        transmat_prior = hmm_params['transMatPrior']
        n_component = hmm_params['nComponent']
        startprob = hmm_params['startProb']
        startprob_prior = hmm_params['startProbPrior']

        n_mix = gmm_params['nMix']
        covariance_type = gmm_params['covarianceType']
        gmms = gmm_params.get('gmms', None)

        gmm_obj_list = []
        if not gmms:
            gmm_obj_list = None
        else:
            for gmm in gmms:
                gmm_obj = GMM(n_components=gmm.n_components, covariance_type=gmm.covariance_type)
                gmm_obj.covars_ = np.array(gmm.covars_)
                gmm_obj.means_ = np.array(gmm.means_)
                gmm_obj.weights_ = np.array(gmm.weights_)
                gmm_obj_list.append(gmm_obj)

        self._model = _model
        self.gmmhmm = GMMHMM(n_components=n_component, n_mix=n_mix, gmms=gmm_obj_list,
                             n_iter=n_iter, covariance_type=covariance_type,
                             transmat=transmat, transmat_prior=transmat_prior,
                             startprob=startprob, startprob_prior=startprob_prior)
        self.params_ = None
        self.train_data_ = []

    def __repr__(self):
        return '<GMMHMMTrainer instance>\n\tinit_models:%s\n\tparams:%s\n\ttrain_data:%s' % (self._model,
                                                                                         self.params_, self.train_data_)

    def fit(self, train_data):
        train_data = np.array(train_data)
        self.gmmhmm.fit(train_data)

        gmms_ = []
        for gmm in self.gmmhmm.gmms_:
            gmms_.append({
                'nComponent': gmm.n_components,
                'nIter': gmm.n_iter,
                'means': gmm.means_,
                'covars': gmm.covars_,
                'weights': gmm.weights_,
                'covariance_type': gmm.covariance_type,
            })
        self.train_data_ += train_data.tolist()
        self.params_ = {
            'nIter': self.gmmhmm.n_iter,
            'hmmParams': {
                'nComponent': self.gmmhmm.n_components,
                'transMat': self.gmmhmm.transmat_,
                'transMatPrior': self.gmmhmm.transmat_prior,
                'startProb': self.gmmhmm.startprob_,
                'startProbPrior': self.gmmhmm.startprob_prior,
            },
            'gmmParams': {
                'nMix': self.gmmhmm.n_mix,
                'covariance_type': self.gmmhmm.covariance_type,
                'gmms': gmms_,
            }
        }


if __name__ == '__main__':
    from datasets import Dataset
    d = Dataset()
    d.randomObservations("dining.chineseRestaurant", 10, 10)

    _model = {
        "hmmParams": {
            "transMat": [
                [
                    0.2,
                    0.1,
                    0.3,
                    0.4
                ],
                [
                    0.3,
                    0.2,
                    0.2,
                    0.3
                ],
                [
                    0.1,
                    0.1,
                    0.1,
                    0.7
                ],
                [
                    0.1,
                    0.3,
                    0.4,
                    0.2
                ]
            ],
            "nComponent": 4,
            "startProb": [
                0.4,
                0.3,
                0.1,
                0.2
            ],
            "startProbPrior": 0.4,
            "transMatPrior": 1.0,
        },
        "gmmParams": {
            "nMix": 4,
            "covarianceType": "full"
        }
    }
    my_trainer = GMMHMMTrainer(_model)
    print(my_trainer)
    print('dataset: %s' % (d.getDataset()))
    my_trainer.fit(d.getDataset())
    print(my_trainer)
