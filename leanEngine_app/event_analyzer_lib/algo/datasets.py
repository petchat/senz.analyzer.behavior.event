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
    event_type = ('travel_in_scenic', 'emergency', 'work_in_office', 'go_for_concert', 'dining_in_restaurant',
                  'exercise_indoor', 'go_for_outing', 'exercise_outdoor', 'go_to_class', 'go_home', 'shopping_in_mall',
                  'movie_in_cinema', 'go_for_exhibition', 'go_work')

    # motion_type = ("sitting", "walking", "running", "riding", "driving")
    motion_type = ('walking', 'driving', 'sitting', 'unknown', 'running', 'riding')

    '''
    sound_type = (
        "keyboard", "bird", "tree", "car_crash", "music", "turning_page", "gun", "talking", "quarrel", "mouse_click",
        "writing", "car_whistle", "school_bell", "wind", "car_driving_by", "stair", "tableware", "laugh", "others",
        "scream", "subway", "boom", "flowing", "speech", "step", "sea", "car_brakes"
    )
    '''
    sound_type = ('shop', 'hallway', 'busy_street', 'quiet_street', 'flat', 'unknown', 'train_station', 'bedroom',
                  'living_room', 'supermarket', 'walk', 'bus_stop', 'classroom', 'subway', 'in_bus',
                  'study_quite_office', 'forrest', 'kitchen')

    '''
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
    '''

    location_type = ('economy_hotel', 'outdoor', 'bath_sauna', 'technical_school', 'bike_store', 'pet_service',
                     'clinic', 'motorcycle', 'guest_house', 'ticket_agent', 'chinese_restaurant', 'flea_market',
                     'resort', 'pet_market', 'digital_store', 'coffee', 'dessert', 'cosmetics_store', 'traffic',
                     'work_office', 'bank', 'adult_education', 'bar', 'talent_market', 'university', 'cooler',
                     'convenience_store', 'snack_bar', 'home', 'post_office', 'hostel', 'motel', 'welfare_house',
                     'farmers_market', 'vegetarian_diet', 'high_school', 'sports_store', 'gas_station',
                     'training_institutions', 'muslim', 'supermarket', 'insurance_company', 'others', 'auto_sale',
                     'video_store', 'commodity_market', 'chafing_dish', 'housekeeping', 'residence',
                     'convention_center', 'atm', 'lottery_station', 'business_building', 'internet_bar', 'mother_store',
                     'museum', 'night_club', 'antique_store', 'japan_korea_restaurant', 'other_infrastructure',
                     'car_maintenance', 'odeum', 'unknown', 'hospital', 'primary_school', 'photographic_studio',
                     'drugstore', 'glass_store', 'bbq', 'auto_repair', 'toll_station', 'hotel', 'newstand', 'stationer',
                     'public_utilities', 'library', 'security_company', 'comprehensive_market', 'salvage_station',
                     'ktv', 'exhibition_hall', 'barbershop', 'clothing_store', 'water_supply_office', 'telecom_offices',
                     'furniture_store', 'gift_store', 'cinema', 'car_wash', 'travel_agency', 'photography_store',
                     'electricity_office', 'pawnshop', 'game_room', 'kinder_garten', 'emergency_center', 'intermediary',
                     'jewelry_store', 'parking_plot', 'laundry', 'scenic_spot', 'buffet', 'gallery',
                     'western_restaurant', 'science_museum', 'seafood', 'cigarette_store')

    rawdata_map = {
        "motion": motion_type,
        "sound": sound_type,
        "location": location_type
    }

    event_prob_map = {
        "go_work": {
            "motion": [
                {
                    "running": 0.1
                },
                {
                    "riding": 0.1
                },
                {
                    "walking": 0.3
                },
                {
                    "sitting": 0.2
                },
                {
                    "driving": 0.3
                }
            ],
            "sound": [
                {
                    "walk": 0.1
                },
                {
                    "bus_stop": 0.1
                },
                {
                    "quiet_street": 0.1
                },
                {
                    "subway": 0.2
                },
                {
                    "in_bus": 0.2
                },
                {
                    "busy_street": 0.3
                }
            ],
            "location": [
                {
                    "traffic": 0.26
                },
                {
                    "Others": 0.1
                },
                {
                    "residence": 0.1
                },
                {
                    "home": 0.05
                },
                {
                    "cigarette_store": 0.01
                },
                {
                    "newstand": 0.05
                },
                {
                    "coffee": 0.05
                },
                {
                    "gas_station": 0.05
                },
                {
                    "university": 0.01
                },
                {
                    "primary_school": 0.01
                },
                {
                    "motorcycle": 0.05
                },
                {
                    "outdoor": 0.1
                },
                {
                    "post_office": 0.01
                },
                {
                    "convenience_store": 0.04
                },
                {
                    "parking_plot": 0.1
                },
                {
                    "toll_station": 0.01
                }
            ]
        },
        "go_to_class": {
            "motion": [
                {
                    "walking": 0.1
                },
                {
                    "sitting": 0.9
                }
            ],
            "sound": [
                {
                    "walk": 0.1
                },
                {
                    "unknown": 0.1
                },
                {
                    "classroom": 0.8
                }
            ],
            "location": [
                {
                    "high_school": 0.1
                },
                {
                    "university": 0.23
                },
                {
                    "primary_school": 0.05
                },
                {
                    "adult_education": 0.1
                },
                {
                    "technical_school": 0.3
                },
                {
                    "science_museum": 0.05
                },
                {
                    "library": 0.05
                },
                {
                    "kinder_garten": 0.01
                },
                {
                    "emergency_center": 0.01
                },
                {
                    "museum": 0.05
                },
                {
                    "training_institutions": 0.05
                }
            ]
        },
        "go_for_concert": {
            "motion": [
                {
                    "walking": 0.1
                },
                {
                    "sitting": 0.9
                }
            ],
            "sound": [
                {
                    "Others": 1
                }
            ],
            "location": [
                {
                    "gallery": 0.2
                },
                {
                    "exhibition_hall": 0.2
                },
                {
                    "odeum": 0.2
                },
                {
                    "motel": 0.05
                },
                {
                    "scenic_spot": 0.05
                },
                {
                    "bar": 0.05
                },
                {
                    "night_club": 0.05
                },
                {
                    "university": 0.1
                },
                {
                    "outdoor": 0.05
                },
                {
                    "ticket_agent": 0.05
                }
            ]
        },
        "travel_in_scenic": {
            "motion": [
                {
                    "running": 0.1
                },
                {
                    "walking": 0.3
                },
                {
                    "riding": 0.1
                },
                {
                    "sitting": 0.2
                },
                {
                    "driving": 0.3
                }
            ],
            "sound": [
                {
                    "walk": 0.1
                },
                {
                    "quiet_street": 0.1
                },
                {
                    "busy_street": 0.2
                },
                {
                    "shop": 0.2
                },
                {
                    "forrest": 0.3
                },
                {
                    "unknown": 0.1
                }
            ],
            "location": [
                {
                    "traffic": 0.2
                },
                {
                    "Others": 0.1
                },
                {
                    "residence": 0.1
                },
                {
                    "scenic_spot": 0.2
                },
                {
                    "museum": 0.1
                },
                {
                    "atm": 0.01
                },
                {
                    "parking_plot": 0.02
                },
                {
                    "emergency_center": 0.01
                },
                {
                    "cooler": 0.01
                },
                {
                    "dessert": 0.01
                },
                {
                    "economy_hotel": 0.01
                },
                {
                    "jewelry_store": 0.01
                },
                {
                    "gallery": 0.01
                },
                {
                    "snack_bar": 0.01
                },
                {
                    "outdoor": 0.05
                },
                {
                    "gas_station": 0.01
                },
                {
                    "muslim": 0.01
                },
                {
                    "travel_agency": 0.01
                },
                {
                    "bank": 0.01
                },
                {
                    "bar": 0.01
                },
                {
                    "coffee": 0.01
                },
                {
                    "hostel": 0.05
                },
                {
                    "toll_station": 0.01
                },
                {
                    "cigarette_store": 0.01
                },
                {
                    "motel": 0.01
                },
                {
                    "drugstore": 0.01
                }
            ]
        },
        "go_home": {
            "motion": [
                {
                    "running": 0.1
                },
                {
                    "walking": 0.3
                },
                {
                    "sitting": 0.2
                },
                {
                    "driving": 0.3
                },
                {
                    "riding": 0.1
                }
            ],
            "sound": [
                {
                    "walk": 0.1
                },
                {
                    "quiet_street": 0.1
                },
                {
                    "subway": 0.2
                },
                {
                    "in_bus": 0.2
                },
                {
                    "busy_street": 0.3
                },
                {
                    "bus_stop": 0.1
                }
            ],
            "location": [
                {
                    "traffic": 0.26
                },
                {
                    "Others": 0.1
                },
                {
                    "residence": 0.1
                },
                {
                    "home": 0.05
                },
                {
                    "cigarette_store": 0.01
                },
                {
                    "newstand": 0.05
                },
                {
                    "coffee": 0.05
                },
                {
                    "gas_station": 0.05
                },
                {
                    "university": 0.01
                },
                {
                    "primary_school": 0.01
                },
                {
                    "motorcycle": 0.05
                },
                {
                    "outdoor": 0.1
                },
                {
                    "post_office": 0.01
                },
                {
                    "convenience_store": 0.04
                },
                {
                    "parking_plot": 0.1
                },
                {
                    "toll_station": 0.01
                }
            ]
        },
        "dining_in_restaurant": {
            "motion": [
                {
                    "walking": 0.2
                },
                {
                    "sitting": 0.8
                }
            ],
            "sound": [
                {
                    "walk": 0.3
                },
                {
                    "kitchen": 0.4
                },
                {
                    "living_room": 0.1
                },
                {
                    "shop": 0.2
                }
            ],
            "location": [
                {
                    "vegetarian_diet": 0.05
                },
                {
                    "western_restaurant": 0.2
                },
                {
                    "chafing_dish": 0.05
                },
                {
                    "buffet": 0.1
                },
                {
                    "muslim": 0.05
                },
                {
                    "seafood": 0.05
                },
                {
                    "bbq": 0.1
                },
                {
                    "chinese_restaurant": 0.2
                },
                {
                    "japan_korea_restaurant": 0.1
                },
                {
                    "coffee": 0.04
                },
                {
                    "dessert": 0.03
                },
                {
                    "scenic_spot": 0.01
                },
                {
                    "bar": 0.01
                },
                {
                    "hotel": 0.01
                },
                {
                    "hostel": 0.01
                }
            ]
        },
        "movie_in_cinema": {
            "motion": [
                {
                    "walking": 0.1
                },
                {
                    "sitting": 0.9
                }
            ],
            "sound": [
                {
                    "Others": 1
                }
            ],
            "location": [
                {
                    "cinema": 0.9
                },
                {
                    "Others": 0.1
                }
            ]
        },
        "emergency": {
            "motion": [
                {
                    "running": 0.5
                },
                {
                    "walking": 0.2
                },
                {
                    "sitting": 0.1
                },
                {
                    "driving": 0.2
                }
            ],
            "sound": [
                {
                    "subway": 0.2
                },
                {
                    "in_bus": 0.3
                },
                {
                    "busy_street": 0.5
                }
            ],
            "location": [
                {
                    "traffic": 0.3
                },
                {
                    "emergency_center": 0.2
                },
                {
                    "residence": 0.1
                },
                {
                    "hospital": 0.2
                },
                {
                    "drugstore": 0.1
                },
                {
                    "salvage_station": 0.1
                }
            ]
        },
        "go_for_outing": {
            "motion": [
                {
                    "running": 0.1
                },
                {
                    "walking": 0.2
                },
                {
                    "sitting": 0.2
                },
                {
                    "driving": 0.4
                },
                {
                    "riding": 0.1
                }
            ],
            "sound": [
                {
                    "walk": 0.1
                },
                {
                    "quiet_street": 0.1
                },
                {
                    "busy_street": 0.2
                },
                {
                    "shop": 0.1
                },
                {
                    "forrest": 0.4
                },
                {
                    "bus_stop": 0.1
                }
            ],
            "location": [
                {
                    "traffic": 0.2
                },
                {
                    "Others": 0.1
                },
                {
                    "residence": 0.1
                },
                {
                    "scenic_spot": 0.2
                },
                {
                    "museum": 0.05
                },
                {
                    "atm": 0.01
                },
                {
                    "parking_plot": 0.02
                },
                {
                    "emergency_center": 0.01
                },
                {
                    "cooler": 0.01
                },
                {
                    "dessert": 0.01
                },
                {
                    "economy_hotel": 0.01
                },
                {
                    "jewelry_store": 0.01
                },
                {
                    "gallery": 0.01
                },
                {
                    "snack_bar": 0.01
                },
                {
                    "outdoor": 0.05
                },
                {
                    "gas_station": 0.01
                },
                {
                    "muslim": 0.01
                },
                {
                    "travel_agency": 0.01
                },
                {
                    "bank": 0.01
                },
                {
                    "bar": 0.01
                },
                {
                    "coffee": 0.01
                },
                {
                    "hostel": 0.05
                },
                {
                    "toll_station": 0.01
                },
                {
                    "cigarette_store": 0.01
                },
                {
                    "motel": 0.01
                },
                {
                    "drugstore": 0.01
                },
                {
                    "resort": 0.05
                }
            ]
        },
        "work_in_office": {
            "motion": [
                {
                    "walking": 0.1
                },
                {
                    "sitting": 0.9
                }
            ],
            "sound": [
                {
                    "walk": 0.1
                },
                {
                    "living_room": 0.2
                },
                {
                    "study_quiet_office": 0.7
                }
            ],
            "location": [
                {
                    "business_building": 0.2
                },
                {
                    "university": 0.1
                },
                {
                    "work_office": 0.3
                },
                {
                    "museum": 0.01
                },
                {
                    "post_office": 0.01
                },
                {
                    "training_institutions": 0.01
                },
                {
                    "water_supply_office": 0.01
                },
                {
                    "telecom_offices": 0.05
                },
                {
                    "adult_education": 0.05
                },
                {
                    "hotel": 0.01
                },
                {
                    "ticket_agent": 0.01
                },
                {
                    "library": 0.01
                },
                {
                    "electricity_office": 0.05
                },
                {
                    "primary_school": 0.01
                },
                {
                    "science_museum": 0.04
                },
                {
                    "technical_school": 0.03
                },
                {
                    "travel_agency": 0.01
                },
                {
                    "bank": 0.04
                },
                {
                    "high_school": 0.03
                },
                {
                    "insurance_company": 0.02
                }
            ]
        },
        "exercise_outdoor": {
            "motion": [
                {
                    "running": 0.4
                },
                {
                    "walking": 0.3
                },
                {
                    "sitting": 0.1
                },
                {
                    "riding": 0.2
                }
            ],
            "sound": [
                {
                    "walk": 0.3
                },
                {
                    "quiet_street": 0.2
                },
                {
                    "busy_street": 0.5
                }
            ],
            "location": [
                {
                    "traffic": 0.3
                },
                {
                    "outdoor": 0.58
                },
                {
                    "university": 0.1
                },
                {
                    "cooler": 0.01
                },
                {
                    "dessert": 0.01
                }
            ]
        },
        "go_for_exhibition": {
            "motion": [
                {
                    "walking": 0.7
                },
                {
                    "sitting": 0.3
                }
            ],
            "sound": [
                {
                    "walk": 0.3
                },
                {
                    "hallway": 0.3
                },
                {
                    "living_room": 0.3
                },
                {
                    "study_quiet_office": 0.1
                }
            ],
            "location": [
                {
                    "museum": 0.3
                },
                {
                    "gallery": 0.2
                },
                {
                    "muslim": 0.1
                },
                {
                    "exhibition_hall": 0.3
                },
                {
                    "university": 0.1
                }
            ]
        },
        "shopping_in_mall": {
            "motion": [
                {
                    "walking": 0.8
                },
                {
                    "sitting": 0.2
                }
            ],
            "sound": [
                {
                    "walk": 0.4
                },
                {
                    "quiet_street": 0.2
                },
                {
                    "busy_street": 0.2
                },
                {
                    "shop": 0.2
                }
            ],
            "location": [
                {
                    "clothing_store": 0.14
                },
                {
                    "sports_store": 0.1
                },
                {
                    "comprehensive_market": 0.1
                },
                {
                    "digital_store": 0.1
                },
                {
                    "cigarette_store": 0.05
                },
                {
                    "video_store": 0.05
                },
                {
                    "dessert": 0.05
                },
                {
                    "pawnshop": 0.01
                },
                {
                    "coffee": 0.05
                },
                {
                    "ktv": 0.01
                },
                {
                    "furniture_store": 0.02
                },
                {
                    "gift_store": 0.05
                },
                {
                    "photography_store": 0.03
                },
                {
                    "cosmetics_store": 0.05
                },
                {
                    "pet_market": 0.05
                },
                {
                    "antique_store": 0.01
                },
                {
                    "commodity_market": 0.01
                },
                {
                    "jewelry_store": 0.1
                },
                {
                    "mother_store": 0.02
                },
                {
                    "supermarket": 0.05
                },
                {
                    "cooler": 0.02
                },
                {
                    "parking_plot": 0.02
                },
                {
                    "atm": 0.01
                }
            ]
        },
        "exercise_indoor": {
            "motion": [
                {
                    "running": 0.1
                },
                {
                    "walking": 0.2
                },
                {
                    "sitting": 0.7
                }
            ],
            "sound": [
                {
                    "walk": 0.2
                },
                {
                    "living_room": 0.8
                }
            ],
            "location": [
                {
                    "home": 0.5
                },
                {
                    "residence": 0.5
                }
            ]
        }
    }

    def __repr__(self):
        return 'Dataset(event_type=%s,\n\tmotion_type=%s,\n\tsound_type=%s,\n\tlocation_type=%s,\n\tevent_prob_map=%s)' % (
            self.event_type, self.motion_type, self.sound_type, self.location_type, self.event_prob_map
        )

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
        # print('result_prob_list: %s'%(results_prob_list))
        for result_prob in results_prob_list:
            # validate input possible result list
            if result_prob.keys()[0] not in self.rawdata_map[rawdata_type] and result_prob.keys()[0] != "Others":
                # print('result_prob:%s' %(result_prob))
                # print('rawdata_map[%s]=%s' % (rawdata_type, self.rawdata_map[rawdata_type]))
                # print('!!!!!!*****  Enterhere  !!!!!*****, raw_type=%s' % (rawdata_type))
                print result_prob.keys()[0]
                return None
                # if result_prob.keys()[0] is "other":
                # result_prob.keys()[0] = ut.chooseRandomly(self.rawdata_map[rawdata_type])
            # print('~~self.rawdata_map: %s' % (self.rawdata_map[rawdata_type]))
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
                # print('fuck JCY, event: %s gogogo' % (event))
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
    dataset.randomObservations("shopping_in_mall", 10, 1)
    print dataset.obs
    print dataset.getDataset()
    print(dataset)
    # dataset.plotObservations3D()

    print(dataset.randomSequence("go_home", 10))
