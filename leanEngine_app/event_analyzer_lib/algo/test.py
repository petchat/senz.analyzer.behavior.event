from hmmlearn.hmm import GMMHMM
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import MultinomialHMM
from sklearn.mixture.gmm import GMM
from datasets import Dataset
import numpy as np

# Prepare parameters for a 3-components HMM
# Initial population probability
start_prob = np.array([0.4, 0.3, 0.1, 0.2])
# The transition matrix, note that there are no transitions possible
# between component 1 and 4
trans_mat = np.array([[0.5, 0.2, 0.2, 0.1],
                      [0.3, 0.4, 0.2, 0.1],
                      [0.1, 0.2, 0.5, 0.2],
                      [0.2, 0.1, 0.1, 0.6]])

start_prob_prior = np.array([0.3, 0.3, 0.3, 0.1])

trans_mat_prior = np.array([[0.2, 0.1, 0.3, 0.4],
                            [0.3, 0.2, 0.2, 0.3],
                            [0.1, 0.1, 0.1, 0.7],
                            [0.1, 0.3, 0.4, 0.2]])

# Build an HMM instance and set parameters
model_dining  = GMMHMM(startprob_prior=start_prob_prior, transmat_prior=trans_mat_prior, startprob=start_prob, transmat=trans_mat, n_components=4, n_mix=4, covariance_type='spherical', n_iter=50)
model_fitness = GMMHMM(startprob_prior=start_prob_prior, transmat_prior=trans_mat_prior, startprob=start_prob, transmat=trans_mat, n_components=4, n_mix=10, covariance_type='spherical', n_iter=50)
model_work    = GMMHMM(startprob_prior=start_prob_prior, transmat_prior=trans_mat_prior, startprob=start_prob, transmat=trans_mat, n_components=4, n_mix=8, covariance_type='spherical', n_iter=50)
model_shop    = GMMHMM(startprob_prior=start_prob_prior, transmat_prior=trans_mat_prior, startprob=start_prob, transmat=trans_mat, n_components=4, n_mix=4, covariance_type='spherical', n_iter=50)

# print model_dining.gmms_[0].covars_.tolist()
# print model_dining.gmms_[0].means_.tolist()
# print model_dining.gmms_[0].weights_.tolist()

dataset_dining  = Dataset()
dataset_fitness = Dataset()
dataset_work    = Dataset()
dataset_shop    = Dataset()

# print Dataset().randomObservations('dining_out_in_chinese_restaurant', 10, 10).obs

D = dataset_dining.randomObservations('dining.chineseRestaurant', 10, 300).getDataset()
F = dataset_fitness.randomObservations('fitness.running', 10, 300).getDataset()
W = dataset_work.randomObservations('work.office', 10, 300).getDataset()
S = dataset_shop.randomObservations('shopping.mall', 10, 300).getDataset()
# dataset_dining.plotObservations3D()

# D = Dataset(obs_dining).dataset
# F = Dataset(obs_fitness).dataset
# W = Dataset(obs_work).dataset
# S = Dataset(obs_shop).dataset

print 'Before training'

print model_dining.startprob_
print model_dining.transmat_
print model_dining.gmms
print model_dining.covariance_type
print model_dining.covars_prior


seq_d_s = dataset_dining.randomSequence('dining.chineseRestaurant', 5)
print 'dining s:'
print seq_d_s
seq_d = dataset_dining.randomSequence('dining.chineseRestaurant', 10)
print 'dining l:'
print seq_d
seq_f = dataset_fitness.randomSequence('fitness.running', 5)
print 'fitness'
print seq_f
seq_w = dataset_work.randomSequence('work.office', 5)
print 'work'
print seq_w
seq_s = dataset_shop.randomSequence('shopping.mall', 5)
print 'shopping'
print seq_s



model_dining.fit(D)
model_fitness.fit(F)
model_work.fit(W)
model_shop.fit(S)


print model_dining.startprob_.tolist()
print model_dining.transmat_.tolist()


print 'After training'

print ' - Classification for seq dining s-'

print 'dining result:'
print model_dining.score(np.array(dataset_dining._convetNumericalSequence(seq_d_s)))
print 'fitness result:'
print model_fitness.score(np.array(dataset_dining._convetNumericalSequence(seq_d_s)))
print 'shop result:'
print model_shop.score(np.array(dataset_dining._convetNumericalSequence(seq_d_s)))
print 'work result:'
print model_work.score(np.array(dataset_dining._convetNumericalSequence(seq_d_s)))

print ' - Classification for seq dining l-'

print 'dining result:'
print model_dining.score(np.array(dataset_dining._convetNumericalSequence(seq_d)))
print 'fitness result:'
print model_fitness.score(np.array(dataset_dining._convetNumericalSequence(seq_d)))
print 'work result:'
print model_work.score(np.array(dataset_dining._convetNumericalSequence(seq_d)))
print 'shop result:'
print model_shop.score(np.array(dataset_dining._convetNumericalSequence(seq_d)))

print ' - Classification for seq fitness-'

print 'dining result'
print model_dining.score(np.array(dataset_fitness._convetNumericalSequence(seq_f)))
print 'fitness result'
print model_fitness.score(np.array(dataset_fitness._convetNumericalSequence(seq_f)))
print 'work result'
print model_work.score(np.array(dataset_fitness._convetNumericalSequence(seq_f)))
print 'shop result'
print model_shop.score(np.array(dataset_fitness._convetNumericalSequence(seq_f)))

print ' - Classification for seq work-'

print 'dining result'
print model_dining.score(np.array(dataset_fitness._convetNumericalSequence(seq_w)))
print 'fitness result'
print model_fitness.score(np.array(dataset_fitness._convetNumericalSequence(seq_w)))
print 'work result'
print model_work.score(np.array(dataset_fitness._convetNumericalSequence(seq_w)))
print 'shop result'
print model_shop.score(np.array(dataset_fitness._convetNumericalSequence(seq_w)))

print ' - Classification for seq shop -'

print 'dining result'
print model_dining.score(np.array(dataset_fitness._convetNumericalSequence(seq_s)))
print 'fitness result'
print model_fitness.score(np.array(dataset_fitness._convetNumericalSequence(seq_s)))
print 'work result'
print model_work.score(np.array(dataset_fitness._convetNumericalSequence(seq_s)))
print 'shop result'
print model_shop.score(np.array(dataset_fitness._convetNumericalSequence(seq_s)))



print model_dining.transmat_.tolist()
print model_dining.startprob_.tolist()
print model_dining.covariance_type
print model_dining.covars_prior
print model_dining.gmms_[0].covars_.tolist()
print model_dining.gmms_[0].means_.tolist()
print model_dining.gmms_[0].weights_.tolist()

print model_dining.gmms_[1].covars_.tolist()
print model_dining.gmms_[1].means_.tolist()
print model_dining.gmms_[1].weights_.tolist()

print model_dining.gmms_[2].covars_.tolist()
print model_dining.gmms_[2].means_.tolist()
print model_dining.gmms_[2].weights_.tolist()

print model_dining.gmms_[3].covars_.tolist()
print model_dining.gmms_[3].means_.tolist()
print model_dining.gmms_[3].weights_.tolist()

# print len(model_dining.gmms_)
#
# gmm = GMM(n_components=4, covariance_type='spherical', random_state=None, thresh=None, tol=1e-3, min_covar=1e-3, n_iter=100, n_init=1, params='wmc', init_params='wmc')
# gmm.covars_  = model_dining.gmms_[0].covars_
# gmm.means_   = model_dining.gmms_[0].means_
# gmm.weights_ = model_dining.gmms_[0].weights_

# {"covars": [[[0.20272925448170726, -0.21372410825705002, 0.6640076953697146],
#              [-0.21372410825704996, 44.41361612877133, -3.8315142667592568],
#              [0.6640076953697147, -3.831514266759256, 29.058072351569287]],
#             [[0.19961779798118065, -0.20567068761490478, 0.6180902066504756],
#              [-0.2056706876149048, 24.5623976379672, -1.8862253825381476],
#              [0.6180902066504755, -1.886225382538147, 28.57872805315829]],
#             [[0.21297430879173704, -0.12933325429515627, 0.6502230892315992],
#              [-0.12933325429515624, 25.319590716568648, -1.135216161881697],
#              [0.6502230892315992, -1.135216161881697, 28.002589075295308]],
#             [[0.20772196929306083, -0.16035047001514102, 0.6106797655515614],
#              [-0.16035047001514105, 26.54555785898669, -1.643554371696736],
#              [0.6106797655515614, -1.643554371696736, 28.157741030750508]]],
#  "weights": [0.24391800630360175, 0.2623161644960391, 0.2516074177870681, 0.242158411413291],
#  "means": [[0.3093785189118171, 98.50767974689373, 12.04201267494459],
#            [0.2957223232822461, 100.29260369087528, 11.368513289910661],
#            [0.32869103266615624, 100.24202740519003, 12.299965535058508],
#            [0.31513728410310543, 100.08656506960568, 12.162225242103984]]}, {"covars": [
#     [[0.20272925448170726, -0.21372410825705002, 0.6640076953697146],
#      [-0.21372410825704996, 44.41361612877133, -3.8315142667592568],
#      [0.6640076953697147, -3.831514266759256, 29.058072351569287]],
#     [[0.19961779798118065, -0.20567068761490478, 0.6180902066504756],
#      [-0.2056706876149048, 24.5623976379672, -1.8862253825381476],
#      [0.6180902066504755, -1.886225382538147, 28.57872805315829]],
#     [[0.21297430879173704, -0.12933325429515627, 0.6502230892315992],
#      [-0.12933325429515624, 25.319590716568648, -1.135216161881697],
#      [0.6502230892315992, -1.135216161881697, 28.002589075295308]],
#     [[0.20772196929306083, -0.16035047001514102, 0.6106797655515614],
#      [-0.16035047001514105, 26.54555785898669, -1.643554371696736],
#      [0.6106797655515614, -1.643554371696736, 28.157741030750508]]], "weights": [0.24391800630360175,
#                                                                                  0.2623161644960391, 0.2516074177870681,
#                                                                                  0.242158411413291], "means": [
#     [0.3093785189118171, 98.50767974689373, 12.04201267494459],
#     [0.2957223232822461, 100.29260369087528, 11.368513289910661],
#     [0.32869103266615624, 100.24202740519003, 12.299965535058508],
#     [0.31513728410310543, 100.08656506960568, 12.162225242103984]]}, {"covars": [
#     [[0.20272925448170726, -0.21372410825705002, 0.6640076953697146],
#      [-0.21372410825704996, 44.41361612877133, -3.8315142667592568],
#      [0.6640076953697147, -3.831514266759256, 29.058072351569287]],
#     [[0.19961779798118065, -0.20567068761490478, 0.6180902066504756],
#      [-0.2056706876149048, 24.5623976379672, -1.8862253825381476],
#      [0.6180902066504755, -1.886225382538147, 28.57872805315829]],
#     [[0.21297430879173704, -0.12933325429515627, 0.6502230892315992],
#      [-0.12933325429515624, 25.319590716568648, -1.135216161881697],
#      [0.6502230892315992, -1.135216161881697, 28.002589075295308]],
#     [[0.20772196929306083, -0.16035047001514102, 0.6106797655515614],
#      [-0.16035047001514105, 26.54555785898669, -1.643554371696736],
#      [0.6106797655515614, -1.643554371696736, 28.157741030750508]]], "weights": [0.24391800630360175,
#                                                                                  0.2623161644960391, 0.2516074177870681,
#                                                                                  0.242158411413291], "means": [
#     [0.3093785189118171, 98.50767974689373, 12.04201267494459],
#     [0.2957223232822461, 100.29260369087528, 11.368513289910661],
#     [0.32869103266615624, 100.24202740519003, 12.299965535058508],
#     [0.31513728410310543, 100.08656506960568, 12.162225242103984]]}, {"covars": [
#     [[0.20272925448170726, -0.21372410825705002, 0.6640076953697146],
#      [-0.21372410825704996, 44.41361612877133, -3.8315142667592568],
#      [0.6640076953697147, -3.831514266759256, 29.058072351569287]],
#     [[0.19961779798118065, -0.20567068761490478, 0.6180902066504756],
#      [-0.2056706876149048, 24.5623976379672, -1.8862253825381476],
#      [0.6180902066504755, -1.886225382538147, 28.57872805315829]],
#     [[0.21297430879173704, -0.12933325429515627, 0.6502230892315992],
#      [-0.12933325429515624, 25.319590716568648, -1.135216161881697],
#      [0.6502230892315992, -1.135216161881697, 28.002589075295308]],
#     [[0.20772196929306083, -0.16035047001514102, 0.6106797655515614],
#      [-0.16035047001514105, 26.54555785898669, -1.643554371696736],
#      [0.6106797655515614, -1.643554371696736, 28.157741030750508]]], "weights": [0.24391800630360175,
#                                                                                  0.2623161644960391, 0.2516074177870681,
#                                                                                  0.242158411413291], "means": [
#     [0.3093785189118171, 98.50767974689373, 12.04201267494459],
#     [0.2957223232822461, 100.29260369087528, 11.368513289910661],
#     [0.32869103266615624, 100.24202740519003, 12.299965535058508],
#     [0.31513728410310543, 100.08656506960568, 12.162225242103984]]}]},
