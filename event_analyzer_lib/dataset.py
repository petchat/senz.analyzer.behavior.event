import numpy as np
import utils as ut
# import matplotlib.pyplot as plt


class Dataset(object):
    """
    Dataset

    private attributes:
        - obs:
        - dataset:

    """
    # rawdata_type   = []
    event_type     = []
    motion_type    = []
    sound_type     = []
    location_type  = []
    event_prob_map = []
    # rawdata_map    = {
    #     "motion": motion_type,
    #     "sound": sound_type,
    #     "location": location_type
    # }

    def __init__(self, event_type, motion_type, sound_type,
                 location_type, event_prob_map):
        # self.rawdata_type = rawdata_type
        self.event_type = event_type
        self.motion_type = motion_type
        self.sound_type = sound_type
        self.location_type = location_type
        self.event_prob_map = event_prob_map
        # self.rawdata_map = {
        #     "motion": self.motion_type,
        #     "sound": self.sound_type,
        #     "location": self.location_type
        # }

    def _convertNumericalObservation(self, obs):
        """
        Convert Numerical Observation

        Convert Observation from Object to numerical python array (ie. list)
        """
        # compose the dataset in numpy array
        numerical_obs = []
        for seq in obs:
            spots_set = []
            for senz in seq:
                spot = [
                    self.motion_type.index(senz["motion"]),
                    self.location_type.index(senz["location"]),
                    self.sound_type.index(senz["sound"])
                ]
                spots_set.append(spot)
            numerical_obs.append(spots_set)
        return numerical_obs

    def _convetNumericalSequence(self, seq):
        """
        Convert Numerical Sequence

        Convert Sequence from Object to numercial python array (ie. list)
        """
        numerical_seq = []
        for senz in seq:
            spot = [
                self.motion_type.index(senz["motion"]),
                self.location_type.index(senz["location"]),
                self.sound_type.index(senz["sound"])
            ]
            numerical_seq.append(spot)
        return numerical_seq

    # Generate fitable dataset
    def _convertObs2Dataset(self, obs):
        """
        Fit Dataset

        Generate dataset which could be fitted by hmmlearn module.
        obs look like:
        [np.array([...]), ...]
        """
        # compose the dataset in numpy array
        dataset = []
        for seq in obs:
            spots_set = []
            for senz in seq:
                spot = [
                    self.motion_type.index(senz["motion"]),
                    self.location_type.index(senz["location"]),
                    self.sound_type.index(senz["sound"])
                ]
                spots_set.append(spot)
            dataset.append(np.array(spots_set))
        return dataset

    # Generate Rawdata in a discrete pdf randomly.
    def _generateRawdataRandomly(self, rawdata_type, results_prob_list):
        """
        Generate Rawdata Randomly

        Generate rawdata randomly in a specifid discrete probability distribution.
        You should choose which rawdata type you want to generate, then
        you need specify the probability distribution by results_prob_list,
        it"s format looks like:
            [{key: prob1}, {key: prob2}, ...]
        """
        # validate input rawdata type
        if rawdata_type not in self.rawdata_type:
            return None
        for result_prob in results_prob_list:
            # validate input possible result list
            if result_prob.keys()[0] not in self.rawdata_map[rawdata_type] and result_prob.keys()[0] != "Others":
                return None
                # if result_prob.keys()[0] is "other":
                # result_prob.keys()[0] = ut.chooseRandomly(self.rawdata_map[rawdata_type])
        results_prob_list = ut.selectOtherRandomly(results_prob_list, self.rawdata_map[rawdata_type])
        # According to results" probability list,
        # generate the random rawdata.
        return ut.discreteSpecifiedRand(results_prob_list)

    # Generate fitabel dataset randomly
    def randomSequence(self, event, length):
        """
        Random Sequence

        Generate a sequence which contains several senzes.
        the sequence can be scored by invoking hmmlearns" score() method after convert to np.array
        You should specify which event you want to generate,
        and length of sequence.
        """
        # validate input event.
        if event not in self.event_prob_map.keys():
            return None
        # generate senz one by one.
        seq = []
        times = 0
        while times < length:
            senz = {}
            # generate every type of rawdata.
            for rawdata_type in self.rawdata_type:
                senz[rawdata_type] = self._generateRawdataRandomly(rawdata_type,
                                                                   self.event_prob_map[event][rawdata_type])
            seq.append(senz)
            times += 1
        # self.seq = seq
        return seq

    def randomObservations(self, event, seq_length, seq_count):
        """
        Random Observations

        Generate an Observations which contains several sequences.
        the observations can be converted to dataset by invoking _convertObs2Dataset.
        You should specify which event you want to generate,
        and every sequence length, and count of sequences.
        """
        # validate input event.
        if event not in self.event_prob_map.keys():
            return None
        obs = []
        times = 0
        while times < seq_count:
            obs.append(self.randomSequence(event, seq_length))
            times += 1
        self.obs = obs
        return self

    # def plotObservations3D(self):
    #     # TODO: Currently we only process 3-dimensinal plot.
    #     # Create a new figure
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     n_obs = np.array(self._convertNumericalObservation(self.obs))
    #     dim = n_obs.shape[0] * n_obs.shape[1]
    #     # Extract every axis data.
    #     xs = n_obs[:, :, 0].reshape(dim, )
    #     ys = n_obs[:, :, 1].reshape(dim, )
    #     zs = n_obs[:, :, 2].reshape(dim, )
    #     # plot
    #     ax.scatter(xs, ys, zs, c="r", marker="o")
    #     ax.set_xlabel("Motion Label")
    #     ax.set_ylabel("Location Label")
    #     ax.set_zlabel("Sound Label")
    #     # show the figure
    #     plt.show()


if __name__ == "__main__":
    dataset = Dataset()
    dataset.randomObservations("dining_out_in_chinese_restaurant", 10, 100)
    print dataset.obs
    # dataset.plotObservations3D()

