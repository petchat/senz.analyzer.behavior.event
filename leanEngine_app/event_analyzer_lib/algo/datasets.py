import numpy as np
import utils as ut
import numpy as np


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
    event_type = ('travel_in_scenic', 'emergency', 'work_in_office', 'go_for_concert', 'dining_in_restaurant',
                  'exercise_indoor', 'go_for_outing', 'exercise_outdoor', 'go_to_class', 'go_home', 'shopping_in_mall',
                  'movie_in_cinema', 'go_for_exhibition', 'go_work')

    # motion_type = ("sitting", "walking", "running", "riding", "driving")
    motion_type = ('walking', 'driving', 'sitting', 'unknown', 'running', 'riding')

    sound_type = ('shop', 'hallway', 'busy_street', 'quiet_street', 'flat', 'unknown', 'train_station', 'bedroom',
                  'living_room', 'supermarket', 'walk', 'bus_stop', 'classroom', 'subway', 'in_bus',
                  'study_quite_office', 'forrest', 'kitchen')

    location_type = ['travel_agency', 'ticket_agent', 'ticket_agent_plane', 'ticket_agent_train', 'post_office', 'telecom_offices', 'telecom_offices_unicom', 'telecom_offices_netcom', 'newstand', 'water_supply_office', 'electricity_office', 'photographic_studio', 'laundry', 'talent_market', 'lottery_station', 'housekeeping', 'housekeeping_lock', 'housekeeping_hour', 'housekeeping_water_deliver', 'intermediary', 'pet_service', 'salvage_station', 'welfare_house', 'barbershop', 'laundry', 'ticket_agent_coach', 'housekeeping_nanny', 'housekeeping_house_moving', 'telecom_offices_tietong', 'ticket_agent_bus', 'telecom_offices_mobile', 'housekeeping_alliance_repair', 'telecom_offices_telecom', 'public_utilities', 'toll_station', 'other_infrastructure', 'public_phone', 'factory', 'city_square', 'refuge', 'public_toilet', 'church', 'industrial_area', 'comprehensive_market', 'convenience_store', 'supermarket', 'digital_store', 'pet_market', 'furniture_store', 'farmers_market', 'commodity_market', 'flea_market', 'sports_store', 'clothing_store', 'video_store', 'glass_store', 'mother_store', 'jewelry_store', 'cosmetics_store', 'gift_store', 'pawnshop', 'antique_store', 'bike_store', 'cigarette_store', 'stationer', 'motorcycle_sell', 'sports_store', 'shopping_street', 'bank', 'atm', 'insurance_company', 'security_company', 'residence', 'business_building', 'community_center', 'bath_sauna', 'ktv', 'bar', 'coffee', 'night_club', 'cinema', 'odeum', 'resort', 'outdoor', 'game_room', 'internet_bar', 'botanic_garden', 'music_hall', 'movie', 'playground', 'temple', 'aquarium', 'cultural_venues', 'fishing_garden', 'picking_garden', 'cultural_palace', 'memorial_hall', 'park', 'zoo', 'chess_room', 'bathing_beach', 'theater', 'scenic_spot', 'agriculture_forestry_and_fishing_base', 'foreign_institutional', 'government_agency', 'minor_institutions', 'tax_authorities', 'motel', 'hotel', 'economy_hotel', 'guest_house', 'hostel', 'farm_house', 'villa', 'dormitory', 'other_hotel', 'apartment_hotel', 'inn', 'holiday_village', 'gas_station', 'parking_plot', 'auto_sale', 'auto_repair', 'motorcycle', 'car_maintenance', 'car_wash', 'motorcycle_service', 'motorcycle_repair', 'golf', 'skiing', 'sports_venues', 'football_field', 'tennis_court', 'horsemanship', 'race_course', 'basketball_court', 'chinese_restaurant', 'japan_korea_restaurant', 'japan_restaurant', 'korea_restaurant', 'western_restaurant', 'bbq', 'chafing_dish', 'seafood_restaurant', 'vegetarian_diet', 'muslim_dish', 'buffet', 'dessert', 'cooler_store', 'snack_bar', 'vegetarian_diet', 'traffic', 'bus_stop', 'subway', 'highway_service_area', 'railway_station', 'airport', 'coach_station', 'traffic_place', 'bus_route', 'subway_track', 'museum', 'exhibition_hall', 'science_museum', 'library', 'gallery', 'convention_center', 'hospital', 'clinic', 'emergency_center', 'drugstore', 'special_hospital', 'home', 'university', 'high_school', 'primary_school', 'kinder_garten', 'training_institutions', 'technical_school', 'adult_education', 'scientific_research_institution', 'driving_school', 'work_office']

    location_one_type = None

    location_map  = {
    'dining': [
        'chinese_restaurant', 'japan_korea_restaurant','japan_restaurant','korea_restaurant', 'western_restaurant', 'bbq', 'chafing_dish', 'seafood_restaurant',
        'vegetarian_diet', 'muslim_dish', 'buffet', 'dessert', 'cooler_store', 'snack_bar','vegetarian_diet'
    ],
    'shopping': [
        'comprehensive_market', 'convenience_store', 'supermarket', 'digital_store', 'pet_market', 'furniture_store',
        'farmers_market', 'commodity_market', 'flea_market', 'sports_store', 'clothing_store', 'video_store',
        'glass_store', 'mother_store', 'jewelry_store', 'cosmetics_store', 'gift_store',
        'pawnshop', 'antique_store', 'bike_store', 'cigarette_store', 'stationer','motorcycle_sell','sports_store','shopping_street'
    ],
    'life_service': [
        'travel_agency', 'ticket_agent','ticket_agent_plane', 'ticket_agent_train','post_office', 'telecom_offices' ,'telecom_offices_unicom', 'telecom_offices_netcom','newstand', 'water_supply_office',
        'electricity_office', 'photographic_studio', 'laundry', 'talent_market', 'lottery_station', 'housekeeping','housekeeping_lock','housekeeping_hour','housekeeping_water_deliver',
        'intermediary', 'pet_service', 'salvage_station', 'welfare_house', 'barbershop','laundry','ticket_agent_coach','housekeeping_nanny','housekeeping_house_moving',
        'telecom_offices_tietong','ticket_agent_bus','telecom_offices_mobile','housekeeping_alliance_repair','telecom_offices_telecom'
    ],
    'entertainment': [
        'bath_sauna', 'ktv', 'bar', 'coffee', 'night_club', 'cinema', 'odeum', 'resort', 'outdoor', 'game_room',
        'internet_bar','botanic_garden','music_hall','movie','playground','temple','aquarium','cultural_venues','fishing_garden','picking_garden','cultural_palace',
        'memorial_hall','park','zoo','chess_room','bathing_beach','theater'
    ],
    'sports':[
      'golf','skiing','sports_venues','football_field','tennis_court','horsemanship','race_course','basketball_court'
    ],
    'auto_related': [
        'gas_station', 'parking_plot', 'auto_sale', 'auto_repair', 'motorcycle', 'car_maintenance', 'car_wash','motorcycle_service','motorcycle_repair'
    ],
    'healthcare': [
        'hospital', 'clinic', 'emergency_center', 'drugstore','special_hospital'
    ],
    'hotel': [
        'motel', 'hotel', 'economy_hotel', 'guest_house', 'hostel','farm_house','villa','dormitory','other_hotel','apartment_hotel','inn','holiday_village'
    ],
    'scenic_spot': ['scenic_spot'
                    ],
    'exhibition': [
        'museum', 'exhibition_hall', 'science_museum', 'library', 'gallery', 'convention_center',
    ],
    'education': [
        'university', 'high_school', 'primary_school', 'kinder_garten', 'training_institutions', 'technical_school',
        'adult_education','scientific_research_institution','driving_school'
    ],
    'finance': [
        'bank', 'atm', 'insurance_company', 'security_company'
    ],
    'infrastructure': [
        'public_utilities', 'toll_station', 'other_infrastructure','public_phone','factory' ,'city_square','refuge','public_toilet','church','industrial_area'
    ],
    'traffic':[
      'traffic','bus_stop','subway','highway_service_area','railway_station','airport','coach_station','traffic_place','bus_route','subway_track'
    ],
    'government':[
        'agriculture_forestry_and_fishing_base','foreign_institutional','government_agency','minor_institutions','tax_authorities'
    ],
    'estate': [
        'residence', 'business_building','community_center'
    ],
    'home': ['home'],
    'work_office': ['work_office'],
}


    rawdata_map = {
        "motion": motion_type,
        "sound": sound_type,
        "location": location_type
    }

    event_prob_map = {
        "attend_concert": {
            "motion": {
                "running": 0.1,
                "walking": 0.1,
                "sitting": 0.6,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.9,
                "possibility": {
                    "gallery": 0.2,
                    "odeum": 0.1,
                    "bar": 0.05,
                    "night_club": 0.05,
                    "cultural_venues": 0.1,
                    "cultural_palace": 0.1,
                    "theater": 0.1,
                    "music_hall": 0.2,
                    "outdoor": 0.1
                }
            }
        },
        "go_outing": {
            "motion": {
                "running": 0.1,
                "riding": 0.1,
                "walking": 0.3,
                "sitting": 0.1,
                "driving": 0.2,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.5,
                "possibility": {
                    "traffic": 0.2,
                    "residence": 0.1,
                    "scenic_spot": 0.2,
                    "museum": 0.05,
                    "atm": 0.01,
                    "parking_plot": 0.02,
                    "emergency_center": 0.01,
                    "cooler_store": 0.01,
                    "dessert": 0.01,
                    "economy_hotel": 0.01,
                    "jewelry_store": 0.01,
                    "gallery": 0.01,
                    "snack_bar": 0.01,
                    "outdoor": 0.05,
                    "gas_station": 0.01,
                    "muslim_dish": 0.01,
                    "travel_agency": 0.01,
                    "bank": 0.01,
                    "bar": 0.01,
                    "coffee": 0.01,
                    "hostel": 0.05,
                    "toll_station": 0.01,
                    "cigarette_store": 0.01,
                    "drugstore": 0.01,
                    "resort": 0.05,
                    "motorcycle": 0.11
                }
            }
        },
        "dining_in_restaurant": {
            "motion": {
                "walking": 0.2,
                "sitting": 0.6,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.9,
                "possibility": {
                    "japan_korea_restaurant": 0.1,
                    "western_restaurant": 0.1,
                    "scenic_spot": 0.05,
                    "chafing_dish": 0.1,
                    "buffet": 0.1,
                    "coffee": 0.05,
                    "bar": 0.05,
                    "vegetarian_diet": 0.05,
                    "dessert": 0.05,
                    "bbq": 0.1,
                    "seafood_restaurant": 0.05,
                    "muslim_dish": 0.05,
                    "chinese_restaurant": 0.15
                }
            }
        },
        "watch_movie": {
            "motion": {
                "walking": 0.1,
                "sitting": 0.7,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.9,
                "possibility": {
                    "movie": 0.5,
                    "cinema": 0.5
                }
            }
        },
        "study_in_class": {
            "motion": {
                "walking": 0.1,
                "sitting": 0.7,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.8,
                "possibility": {
                    "high_school": 0.1,
                    "university": 0.2,
                    "technical_school": 0.15,
                    "primary_school": 0.05,
                    "library": 0.05,
                    "kinder_garten": 0.05,
                    "adult_education": 0.1,
                    "training_institutions": 0.15,
                    "driving_school": 0.1,
                    "museum": 0.05
                }
            }
        },
        "visit_sights": {
            "motion": {
                "running": 0.1,
                "riding": 0.1,
                "walking": 0.3,
                "sitting": 0.2,
                "driving": 0.1,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.6,
                "possibility": {
                    "picking_garden": 0.05,
                    "memorial_hall": 0.05,
                    "public_phone": 0.05,
                    "japan_korea_restaurant": 0.05,
                    "western_restaurant": 0.05,
                    "scenic_spot": 0.1,
                    "traffic": 0.05,
                    "zoo": 0.05,
                    "exhibition_hall": 0.05,
                    "gift_store": 0.05,
                    "science_museum": 0.05,
                    "outdoor": 0.1,
                    "gallery": 0.05,
                    "fishing_garden": 0.05,
                    "residence": 0.05,
                    "church": 0.05,
                    "park": 0.05,
                    "parking_plot": 0.05
                }
            }
        },
        "work_in_office": {
            "motion": {
                "running": 0.1,
                "walking": 0.2,
                "sitting": 0.5,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.9,
                "possibility": {
                    "business_building": 0.2,
                    "university": 0.1,
                    "work_office": 0.3,
                    "museum": 0.01,
                    "post_office": 0.01,
                    "training_institutions": 0.01,
                    "water_supply_office": 0.01,
                    "telecom_offices": 0.01,
                    "telecom_offices_tietong": 0.01,
                    "telecom_offices_unicom": 0.01,
                    "telecom_offices_netcom": 0.01,
                    "telecom_offices_mobile": 0.01,
                    "telecom_offices_telecom": 0.01,
                    "adult_education": 0.04,
                    "hotel": 0.01,
                    "ticket_agent": 0.01,
                    "library": 0.01,
                    "electricity_office": 0.05,
                    "primary_school": 0.01,
                    "science_museum": 0.01,
                    "technical_school": 0.03,
                    "travel_agency": 0.03,
                    "bank": 0.05,
                    "high_school": 0.01,
                    "insurance_company": 0.04
                }
            }
        },
        "exercise_outdoor": {
            "motion": {
                "running": 0.4,
                "riding": 0.2,
                "walking": 0.1,
                "sitting": 0.1,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.5,
                "possibility": {
                    "traffic": 0.3,
                    "outdoor": 0.38,
                    "university": 0.1,
                    "cooler_store": 0.01,
                    "dessert": 0.01,
                    "playground": 0.2
                }
            }
        },
        "shopping_in_mall": {
            "motion": {
                "running": 0.1,
                "walking": 0.5,
                "sitting": 0.2,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.5,
                "possibility": {
                    "clothing_store": 0.1,
                    "sports_store": 0.06,
                    "comprehensive_market": 0.05,
                    "digital_store": 0.05,
                    "cigarette_store": 0.06,
                    "video_store": 0.05,
                    "dessert": 0.09,
                    "pawnshop": 0.01,
                    "coffee": 0.05,
                    "ktv": 0.01,
                    "furniture_store": 0.02,
                    "gift_store": 0.05,
                    "photographic_studio": 0.02,
                    "cosmetics_store": 0.1,
                    "pet_market": 0.05,
                    "antique_store": 0.01,
                    "commodity_market": 0.01,
                    "jewelry_store": 0.1,
                    "mother_store": 0.02,
                    "supermarket": 0.04,
                    "cooler_store": 0.02,
                    "parking_plot": 0.02,
                    "atm": 0.01
                }
            }
        },
        "exercise_indoor": {
            "motion": {
                "running": 0.1,
                "walking": 0.2,
                "sitting": 0.5,
                "unknown": 0.2
            },
            "sound": {
                "unknown": 1
            },
            "location": {
                "visit_freq": 0.9,
                "possibility": {
                    "home": 0.5,
                    "residence": 0.5
                }
            }
        }
    }

    def __repr__(self):
        return 'Dataset(event_type=%s,\n\tmotion_type=%s,\n\tsound_type=%s,\n\tlocation_type=%s,\n\tevent_prob_map=%s)' % (
            self.event_type, self.motion_type, self.sound_type, self.location_type, self.event_prob_map
        )

    def __init__(self,
                 obs=None,
                 rawdata_type=None,
                 event_type=None,
                 motion_type=None,
                 sound_type=None,
                 location_type=None,
                 location_one_type=None,
                 event_prob_map=None,
                 binary_obs=[]):

        self.obs = obs
        self.binary_obs=  binary_obs

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
        if location_one_type is not None:
            self.location_one_type = location_one_type
        if event_prob_map is not None:
            self.event_prob_map = event_prob_map

        if self.location_one_type != None:
            self.binary_senz_length = len(self.location_one_type)\
                                      + len(self.location_type)\
                                      + len(self.motion_type)\
                                      + len(self.sound_type)

        self.rawdata_map = {
            "motion": self.motion_type,
            "sound": self.sound_type,
            "location": self.location_type
        }

    def _find_location_one(self, location_type_two):
        # this is inefficient, but this is training process,so we can omit this timedealy
        for key,values in self.location_map.items():
            if location_type_two in values:
                return key
        return None

    def convert_binary_sequence(self,obs):

        def senz_2_index(senz):

            assert self.location_one_type != None , \
                "RNNRBM algo must set the right location_one set,but now location_one_type is none"

            return [
                self.motion_type.index(senz["motion"]),
                self.location_type.index(senz["location"]),
                self.location_one_type.index(senz["location_one"]),
                self.sound_type.index(senz["sound"])

            ]

        def index_2_binary(indexs):

            '''

            :param indexs:
            :return:  binary order is [motion, location, location_two,sound]
            '''
            motions_zeros = np.zeros(len(self.motion_type))
            motions_zeros[indexs[0]] = 1
            location_zeros = np.zeros(len(self.location_type))
            location_zeros[indexs[1]] = 1
            location_one_zeros = np.zeros(len(self.location_one_type))
            location_one_zeros[indexs[2]] = 1
            sound_zeros = np.zeros(len(self.sound_type))
            sound_zeros[indexs[3]] = 1

            binary_seq = np.hstack([motions_zeros,location_zeros,location_one_zeros,sound_zeros])
            #print "binary_seq",binary_seq
            #print "ndim",binary_seq.shape
            return  binary_seq


        for seq in obs:
            index_seq = map(senz_2_index,seq)
            binary_seq = map(index_2_binary, index_seq)
            self.binary_obs.append(binary_seq)

        return np.array(self.binary_obs)


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
            raise ValueError, "Type of raw data [%s] is incorrect." % rawdata_type
        for result_prob_key in results_prob_list:
            # validate input possible result list
            if result_prob_key not in self.rawdata_map[rawdata_type] and result_prob_key != "Others":
                raise ValueError, "Format of prob map is incorrect. the key %s is incorrect." % result_prob_key
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
        # First choose a place where intention of user actually locates.
        the_place = self._generateRawdataRandomly("location", self.event_prob_map[event]["location"]["possibility"])
        # Generate counterfeit items of seq one by one.
        while times < length:
            senz = {}
            senz["motion"]   = self._generateRawdataRandomly("motion", self.event_prob_map[event]["motion"])
            senz["sound"]    = "unknown"
            # rebuild the prob map of location.
            new_location_prob_map = {"Others": self.event_prob_map[event]["location"]["visit_freq"]}
            new_location_prob_map[the_place] = 0.8
            senz["location"] = self._generateRawdataRandomly("location", new_location_prob_map)
            if self.location_one_type != None:
                senz["location_one"] = self._find_location_one(senz["location"])
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

if __name__ == "__main__":
    dataset = Dataset()
    dataset.randomObservations("exercise_outdoor", 10, 1)
    for i in dataset.obs[0]:
        print i

    # print(dataset.randomSequence("go_home", 10))
