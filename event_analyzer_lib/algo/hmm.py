from hmmlearn.hmm import _BaseHMM
from hmmlearn.utils import normalize
import numpy as np
import string

NEGINF = -np.inf


class DiscreteHMM(_BaseHMM):
    def __init__(self,
        n_components=1,  # Number of states in the model.

        startprob=None,  # Initial state occupation distribution.
        transmat=None,  # Matrix of transition probabilities between states.

        startprob_prior=None,  # The prior probability of initial state occupation distribution.
        transmat_prior=None,  # The prior probability of matrix of transition probabilities between states.
        algorithm="viterbi",  # Decoder algorithm.

        random_state=None,  # A random number generator instance.
        n_iter=10,  # Number of iterations to perform.
        thresh=1e-2,  # EM threshold.

        params=string.ascii_letters,
        # Controls which parameters are updated in the training process.
        # Can contain any combination of 's' for startprob, 't' for transmat, etc.
        # Defaults to all parameters.
        init_params=string.ascii_letters
        # Controls which parameters are initialized prior to training.
        # Can contain any combination of 's' for startprob, 't' for transmat, etc.
        # Defaults to all parameters.
        ):

        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          thresh=thresh, params=params,
                          init_params=init_params)

    def _get_emissionmat(self):
        """Emission probability distribution for each state."""
        return self._emissionmat_

    def _set_emissionmat(self, emissionprob):
        # Convert list to numpy array.
        emissionprob = np.asarray(emissionprob)
        if hasattr(self, 'n_symbols') and emissionprob.shape != (self.n_components, self.n_symbols):
            raise ValueError('emissionprob must have shape '
                             '(n_components, n_symbols)')

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(emissionprob):
            normalize(emissionprob)
        self._emissionmat_ = emissionprob
        # check if there exists any element whose value is NaN
        underflow_idx = np.isnan(self._emissionmat_)
        # set the NaN value as negative inf.
        self._emissionmat_[underflow_idx] = NEGINF
        self.n_symbols = self._emissionmat_.shape[1]

    # Property: emission matrix
    emissionmat_ = property(_get_emissionmat, _set_emissionmat)

    def _compute_log_likelihood(self, obs):
        # T is transpose.
        return self._emissionmat_[:, obs].T

    def _initialize_sufficient_statistics(self):
        stats = super(DiscreteHMM, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_symbols))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(DiscreteHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            for t, symbol in enumerate(obs):
                stats['obs'][:, symbol] += posteriors[t]


    def _do_mstep(self, stats, params):
        super(DiscreteHMM, self)._do_mstep(stats, params)