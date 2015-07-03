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
    rawdata_type = ("motion", "location", "sound")

    # event_type  = ("shopping", "dining_out_in_chinese_restaurant", "work", "running_fitness")
    event_type = ("shopping.mall", "dining.chineseRestaurant", "work.office", "fitness.running")

    motion_type = ("sitting", "walking", "running", "riding", "driving")

    # sound_level1_type  = ("quiet", "lowish", "noisy")

    sound_type = (
    "keyboard", "bird", "tree", "car_crash", "music", "turning_page", "gun", "talking", "quarrel", "mouse_click",
    "writing", "car_whistle", "school_bell", "wind", "car_driving_by", "stair", "tableware", "laugh", "others",
    "scream", "subway", "boom", "flowing", "speech", "step", "sea", "car_brakes"
    )

    # location_level1_type = ("dining", "shopping", "life_service", "entertainment", "auto_related", "healthcare", "hotel", "scenic_spot", "exhibition", "education", "finance", "infrastructure", "estate")

    location_type = (
        "hospital", "bath_sauna", "drugstore", "vegetarian_diet", "talent_market", "business_building",
        "comprehensive_market", "motel", "stationer", "high_school", "insurance_company", "home", "resort",
        "digital_store", "cigarette_store", "pawnshop", "auto_sale", "japan_korea_restaurant", "toll_station",
        "salvage_station", "newstand", "western_restaurant", "car_maintenance", "scenic_spot", "barbershop",
        "chafing_dish", "buffet", "convenience_store", "odeum", "pet_service", "traffic", "cinema", "coffee",
        "auto_repair", "bar", "hostel", "video_store", "others", "game_room", "laundry", "photographic_studio", "ktv",
        "exhibition_hall", "bank", "night_club", "bike_store", "furniture_store", "travel_agency", "technical_school",
        "welfare_house", "intermediary", "security_company", "gift_store", "muslim", "lottery_station",
        "photography_store", "science_museum", "sports_store", "gas_station", "university", "primary_school", "outdoor",
        "motorcycle", "electricity_office", "library", "conventioncenter", "kinder_garten", "ticket_agent",
        "snack_bar", "hotel", "cosmetics_store", "adult_education", "telecom_offices", "pet_market", "housekeeping",
        "antique_store", "work_office", "seafood", "gallery", "bbq", "water_supply_office", "other_infrastructure",
        "residence", "clinic", "internet_bar", "commodity_market", "guest_house", "clothing_store", "farmers_market",
        "flea_market", "jewelry_store", "training_institutions", "post_office", "mother_store", "supermarket",
        "economy_hotel", "glass_store", "public_utilities", "dessert", "cooler", "emergency_center", "car_wash",
        "parking_plot", "chinese_restaurant", "atm", "museum"
    )

    rawdata_map = {
        "motion": motion_type,
        "sound": sound_type,
        "location": location_type
    }

    event_prob_map = {
        "dining.chineseRestaurant": {
            "motion": [{"sitting": 0.7}, {"walking": 0.3}],
            "sound": [{"talking": 0.4}, {"laugh": 0.3}, {"others": 0.15}, {"tableware": 0.1}, {"Others": 0.05}],
            "location": [{"chinese_restaurant": 0.7}, {"Others": 0.25}, {"residence": 0.05}]
        },
        "fitness.running": {
            "motion": [{"running": 0.4}, {"walking": 0.4}, {"sitting": 0.2}],
            "sound": [{"step": 0.2}, {"wind": 0.2}, {"others": 0.2}, {"car_driving_by": 0.2}, {"Others": 0.2}],
            "location": [{"scenic_spot": 0.3}, {"Others": 0.3}, {"residence": 0.2}, {"traffic": 0.2}]
        },
        "work.office": {
            "motion": [{"sitting": 0.85}, {"walking": 0.15}],
            "sound": [{"step": 0.2}, {"others": 0.2}, {"keyboard": 0.2}, {"mouse_click": 0.1}, {"turning_page": 0.1},
                      {"talking": 0.2}],
            "location": [{"work_office": 0.7}, {"home": 0.2}, {"Others": 0.1}]
        },
        "shopping.mall": {
            "motion": [{"walking": 0.7}, {"sitting": 0.3}],
            "sound": [{"step": 0.2}, {"talking": 0.4}, {"car_driving_by": 0.1}, {"music": 0.2}, {"Others": 0.1}],
            "location": [{"comprehensive_market": 0.7}, {"chinese_restaurant": 0.2}, {"Others": 0.1}]
        }
    }

    def __init__(self, obs=None, rawdata_type=None, event_type=None, motion_type=None, sound_type=None,
                 location_type=None, event_prob_map=None):
        self.obs = obs
        if rawdata_type is not None:
            self.rawdata_type = rawdata_type
        if event_type is not None:
            self.event_type = event_type
        if motion_type is not None:
            self.motion_type = motion_type
        if sound_type is not None:
            self.sound_type = sound_type
        if location_type is not None:
            self.location_type = location_type
        if event_prob_map is not None:
            self.event_prob_map = event_prob_map

        self.rawdata_map = {
            "motion": self.motion_type,
            "sound": self.sound_type,
            "location": self.location_type
        }

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

    def getDataset(self):
        """
        Get Dataset

        You can invoke this method to get the dataset,
        which can be fitted by variety of hmm model.
        """
        if self.obs is None:
            return None
        else:
            return self._convertObs2Dataset(self.obs)

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
    dataset.randomObservations("dining.chineseRestaurant", 10, 1)
    print dataset.obs
    print dataset.getDataset()
    # dataset.plotObservations3D()

