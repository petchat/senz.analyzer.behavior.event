from hmmlearn.hmm import GMMHMM
from datasets import Dataset
from sklearn.mixture.gmm import GMM
import numpy as np

def classifyByGMMHMM(seq, models, configs):

    Y = []
    for config in configs:
        _rawdata_type = config["logType"]
        _event_type = config["eventType"]
        _motion_type = config["motionType"]
        _sound_type = config["soundType"]
        _location_type = config["locationType"]

        d = Dataset(
            rawdata_type=_rawdata_type,
            event_type=_event_type,
            motion_type=_motion_type,
            sound_type=_sound_type,
            location_type=_location_type
        )
        # Initiation of data need prediction.
        y = np.array(d._convetNumericalSequence(seq))
        Y.append(y)


    _GMMHMMs = []
    for model in models:
        _GMMs = []
        for gmm in model["gmmParams"]["params"]:
            _GMM = GMM(
                n_components=model["nMix"],
                covariance_type=model["covarianceType"]
            )
            _GMM.covars_  = np.array(gmm["covars"])
            _GMM.means_   = np.array(gmm["means"])
            _GMM.weights_ = np.array(gmm["weights"])
            _GMMs.append(_GMM)
        _GMMHMM = GMMHMM(
            n_components=model["nComponent"],
            n_mix=model["nMix"],
            startprob=np.array(model["hmmParams"]["startProb"]),
            transmat=np.array(model["hmmParams"]["transMat"]),
            gmms=_GMMs,
            covariance_type=model["covarianceType"]
        )
        _GMMHMMs.append(_GMMHMM)

    results = []
    # for _GMMHMM in _GMMHMMs:
        # res = _GMMHMM.score(Y)
        # results.append(res)
    for i in range(0, len(models)):
        res = _GMMHMMs[i].score(Y[i])
        results.append(res)

    return results

class GMMHMMClassifer(object):
    '''A wrapper to a set of GMMHMMs for predict

    Attributes
    ----------
    _models: list of dict
      init models params
    gmmhmms: dict
      keys are labels of GMMHMMs, values are dict {'gmmhmm': instance, 'status_set':status_set}
    predict_data_: current predict_data
    '''

    def __init__(self, _models):
        self._models = _models
        self.gmmhmms = {}
        self.predict_data_ = None

        for label, value in _models.iteritems():
            _model = value['param']
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

            gmmhmm = GMMHMM(n_components=n_component, n_mix=n_mix, gmms=gmm_obj_list,
                            n_iter=n_iter, covariance_type=covariance_type,
                            transmat=transmat, transmat_prior=transmat_prior,
                            startprob=startprob, startprob_prior=startprob_prior)
            self.gmmhmms[label] = {'gmmhmm':gmmhmm, 'status_set':value['status_set']}

    def __repr__(self):
        return '<GMMHMMClassifer instance>\n\tinit_models:%s\n\ttrain_data:%s' % (self._models, self.predict_data_)

    def predict(self, seq):
        result = {}
        self.predict_data_ = seq
        for label, value in self.gmmhmms.iteritems():
            gmmhmm = value['gmmhmm']
            status_set = value['status_set']
            d = Dataset(motion_type=status_set['motion'], sound_type=status_set['sound'],
                        location_type=status_set['location'])
            try:
                seq_converted = np.array(d._convetNumericalSequence(seq))
            except ValueError:
                result[label] = 0.0
            else:
                result[label] = gmmhmm.score(seq_converted)
        return result


# if __name__ == "__main__":
#     seq = [{"motion": "sitting", "sound": "tableware", "location": "chinese_restaurant"}, {"motion": "sitting", "sound": "talking", "location": "chinese_restaurant"}, {"motion": "walking", "sound": "talking", "location": "night_club"}, {"motion": "walking", "sound": "tableware", "location": "chinese_restaurant"}, {"motion": "sitting", "sound": "talking", "location": "night_club"}, {"motion": "sitting", "sound": "laugh", "location": "night_club"}, {"motion": "sitting", "sound": "talking", "location": "night_club"}, {"motion": "walking", "sound": "silence", "location": "chinese_restaurant"}, {"motion": "walking", "sound": "laugh", "location": "chinese_restaurant"}, {"motion": "sitting", "sound": "laugh", "location": "chinese_restaurant"}]
#     models = [
#         {
#             "gmmParams":
#                 {"covarianceType": "spherical",
#                  "nMix": 4,
#                  "params": [
#                      {
#                          "covars": [
#                              [472.27800685, 472.27800685, 472.27800685],
#                              [697.52335729, 697.52335729, 697.52335729],
#                              [530.22124671, 530.22124671, 530.22124671],
#                              [683.69353879, 683.69353879, 683.69353879]
#                          ],
#                          "means": [
#                              [0.31670098, 19.11287179, 10.44514015],
#                              [0.32403931, 44.43047089, 10.69711672],
#                              [0.31640365, 22.64918034, 12.02879255],
#                              [0.3141375, 38.86666949, 11.48627219]
#                          ],
#                          "weights": [0.29628029, 0.23860758, 0.25002084, 0.21509129]
#                      },
#                      {
#                          "covars": [
#                              [471.70382709, 471.70382709, 471.70382709],
#                              [696.80846154, 696.80846154, 696.80846154],
#                              [529.57298416, 529.57298416, 529.57298416],
#                              [683.2172196, 683.2172196, 683.2172196]
#                          ],
#                          "means": [
#                              [0.31844091, 19.09680206, 10.44945422],
#                              [0.32515545, 44.39202715, 10.70965751],
#                              [0.31788919, 22.63183586, 12.03638126],
#                              [0.31535478, 38.85864214, 11.50031672]
#                          ],
#                          "weights": [0.29632258, 0.23851096, 0.25003777, 0.21512869]
#                      },
#                      {
#                          "covars": [
#                              [446.11339906, 446.11339906, 446.11339906],
#                              [677.63354862, 677.63354862, 677.63354862],
#                              [663.83568757, 663.83568757, 663.83568757],
#                              [500.78388024, 500.78388024, 500.78388024]
#                          ],
#                          "means": [
#                              [0.32317726, 17.86576794, 10.65557801],
#                              [0.32433085, 42.17910084, 11.01598356],
#                              [0.31646136, 37.09618443, 11.78900317],
#                              [0.32218379, 21.12328751, 12.31919079]
#                          ],
#                          "weights": [0.29852455, 0.23476512, 0.21422263, 0.2524877]
#                      },
#                      {
#                          "covars": [
#                              [459.68500524, 459.68500524, 459.68500524],
#                              [689.55964417, 689.55964417, 689.55964417],
#                              [516.09011551, 516.09011551, 516.09011551],
#                              [675.41996403, 675.41996403, 675.41996403]
#                          ],
#                          "means": [
#                              [0.32292795, 18.48890516, 10.55491072],
#                              [0.32628212, 43.2975017, 10.87184719],
#                              [0.32187151, 21.88796335, 12.18269494],
#                              [0.31753251, 38.01261467, 11.65670113]
#                          ],
#                          "weights": [0.29741866, 0.23660316, 0.25123148, 0.21474671]
#                      }
#                  ]},
#             "hmmParams": {
#                 "nComponent": 4,
#                 "startProb": [0.41851852, 0.30740741, 0.08518519, 0.18888889],
#                 "transMat": [
#                     [0.50881945, 0.19622023, 0.19874008, 0.09622023],
#                     [0.30335416, 0.40670833, 0.19664584, 0.09329167],
#                     [0.08977197, 0.19488598, 0.51022803, 0.20511402],
#                     [0.1950282, 0.09337093, 0.0950282, 0.61657267]
#                 ]
#             },
#             "eventLabel": "dining_out_in_chinese_restaurant",
#             "nComponent": 4,
#             "nMix": 4,
#             "covarianceType": "full"
#         }
#     ]
#     print classifyByGMMHMM(seq, models)

