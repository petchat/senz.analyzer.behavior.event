# -*- coding: utf-8 -*-

from dao.config import *
from dao.model import *
from algo.datasets import Dataset
import datetime
import logging
import json
from algo import trainer, classifier
from algo.rnnrbm import Comparator


logger = logging.getLogger('logentries')


ALGOMAP = {'GMMHMM': trainer.GMMHMMTrainer}  # algo_type to trainer class map


def rebuildEvent(
        event_type,
        algo_type="GMMHMM",
        new_tag="TAG_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
):
    # Get events' info from db.
    events = getEventInfo()
    # Validation of existance of the event
    if event_type not in events:
        logger.debug("There is no event named %s in Config Class" % event_type)
        return None

    # Get Model's init params.
    init_params = events[event_type]["initParams"][algo_type]
    # # Get Sets of all kinds of status classification.
    sys_status_sets = getSystemStatusSets()

    logger.debug("Event %s's params is %s" % (event_type, init_params))
    logger.debug("And latest status classification are %s" % sys_status_sets)
    logger.debug("The generated model tag is %s" % new_tag)

    now = datetime.datetime.now()
    description = "Initiation of A new %s Model for event %s was made at %s" % (algo_type, event_type, now)
    return setModel(algo_type=algo_type, model_tag=new_tag, event_type=event_type,
                    model_param=init_params, status_sets=sys_status_sets, timestamp=now, description=description)


def initAll(new_tag, algo_type):
    # Get events' info from db.
    events = getEventInfo()

    for event in events:
        rebuildEvent(event, algo_type, new_tag)

    return True


def trainEvent(observations, event_type, source_tag, target_tag, algo_type):
    '''Train model by input observations
    '''
    model = getModel(algo_type, source_tag, event_type)
    model_param = model["modelParam"]
    status_sets = model["statusSets"]

    sound_set = status_sets["sound"]
    motion_set = status_sets["motion"]
    location_set = status_sets["location"]
    d = Dataset(event_type=getEventList(), motion_type=motion_set, sound_type=sound_set,
                location_type=location_set, event_prob_map=getEventProbMap())
    numerical_obs = d._convertObs2Dataset(observations)
    logger.debug('[trainEvent] numerical_obs=%s' % (numerical_obs))

    TRAINER = ALGOMAP[algo_type]
    my_trainer = TRAINER(model_param)
    my_trainer.fit(numerical_obs)
    description = '[source_tag=%s]Train model algo_type=%s for eventType=%s' % (
        source_tag, algo_type, event_type)

    return setModel(algo_type, target_tag, event_type, my_trainer.params_, status_sets,
                    datetime.datetime.now(), description, json.dumps(observations))


def trainEventRandomly(
        event_type,
        source_tag,
        target_tag,
        obs_len,
        obs_count,
        algo_type="GMMHMM"
):
    logger.info('[trainEventRandomly] event_type=%s, source_tag=%s, target_tag=%s'
                % (event_type, source_tag, target_tag))
    model = getModel(algo_type, source_tag, event_type)
    model_param = model["modelParam"]
    status_sets = model["statusSets"]

    sound_set = status_sets["sound"]
    motion_set = status_sets["motion"]
    location_set = status_sets["location"]

    train_obs_len = obs_len
    train_obs_count = obs_count
    d = Dataset(event_type=getEventList(), motion_type=motion_set, sound_type=sound_set,
                location_type=location_set, event_prob_map=getEventProbMap())
    logger.debug('[trainEventRandomly] Dataset: %s' % (d))
    d.randomObservations(event_type, train_obs_len, train_obs_count)   # 这里是产生假序列的部分
    observations = d.obs
    logger.debug('[trainEventRandomly] obs: %s' % (observations))

    description = '[source_tag=%s]Random train algo_type=%s for eventType=%s, random train obs_len=%s, obs_count=%s' % (
        source_tag, algo_type, event_type, train_obs_len, train_obs_count)
    TRAINER = ALGOMAP[algo_type]
    my_trainer = TRAINER(model_param)
    my_trainer.fit(d.getDataset())

    return setModel(algo_type, target_tag, event_type, my_trainer.params_, status_sets,
                    datetime.datetime.now(), description, json.dumps(observations))

def trainRandomRnnRBM():

    #event_type = "dining_in_restaurant"
    train_obs_len = 10  #todo need more than 10 or less than 10 length seqs
    train_obs_count = 10
    event_list = getEventList()
    sound_set = get_sound_set()
    motion_set = get_motion_set()

    location_one_set = get_location_one_set()
    location_two_set = get_location_two_set()
    #the original location type is location_one_type
    d = Dataset(event_type=getEventList(), motion_type=motion_set, sound_type=sound_set,
                location_type=location_two_set, location_one_type=location_one_set, event_prob_map=getEventProbMap())
    #logger.debug('[trainEventRandomly] Dataset: %s' % (d))

    print "dataset",d
    event_dict = {}
    for event_type in event_list: #
        d.randomObservations(event_type, train_obs_len, train_obs_count)   # 这里是产生假序列的部分
        observation_list = d.obs
        binary_sequences = d.convert_binary_sequence(observation_list)
        event_dict.update({event_type: binary_sequences})
    #             (self,
    #              event_params_dict=None, # key is the eventtype ,and the value is the params of RnnRbm,
    #                                      #如果eventtype下有参数，则fit时，是从之前训练的基础上开始训练，如果是空。则是用buildrbmrnn的
    #                                      #default参数来初始化并开始巡礼那
    #              weights_dict=None,
    #              seq_len = 0
    #              ):
    senz_len = len(sound_set) + len(motion_set) + len(location_one_set) + len(location_two_set)

    # def cold_start_params_dict(event_list):
    #     temp_dict = {}
    #     for event in event_list: temp_dict.update({event:None})
    #     return temp_dict

    cmptor = Comparator(senz_len=senz_len,event_list=event_list)
    params_dict = cmptor.fit(event_dict)
    ids = save_rnnrbm_params(params_dict=params_dict,
                             base_set_dict=dict(
                                 motion=motion_set,
                                 location_two=location_two_set,
                                 location_one=location_one_set,
                                 sound=sound_set
                             ),
                             tag="random_v0")
    print "The ids saved  in leancloud are",ids
    print "observations event dict start"
    for i in event_dict:
        print i
    print "observation event dict end"
    return ids


def predictByRnnRbm(seq=None):

    seq = [[
               {
                "location":"economy_hotel",
                "motion":"unknown",
                "location_one": "hotel",
                "sound":"unknown",
                #"tenMinScale":80,
                #"timestamp":1438580199333
                },
                {
                "location":"economy_hotel",
                "motion":"unknown",
                "location_one": "hotel",
                "sound":"unknown",
                #"tenMinScale":80,
                #"timestamp":1438580199333
                },
                {
                "location":"economy_hotel",
                "motion":"unknown",
                "location_one": "hotel",
                "sound":"unknown",
                #"tenMinScale":80,
                #"timestamp":1438580199333
                },
                {
                "location":"economy_hotel",
                "motion":"unknown",
                "location_one": "hotel",
                "sound":"unknown",
                #"tenMinScale":80,
                #"timestamp":1438580199333
                }
    ]]

    event_list = getEventList()
    sound_set = get_sound_set()
    motion_set = get_motion_set()

    location_one_set = get_location_one_set()
    location_two_set = get_location_two_set()
    #the original location type is location_one_type
    d = Dataset(event_type=getEventList(), motion_type=motion_set, sound_type=sound_set,
                location_type=location_two_set, location_one_type=location_one_set, event_prob_map=getEventProbMap())
    binary_sequences = d.convert_binary_sequence(seq)
    event_params_dict = get_all_rnnrbm_params(tag=None,event_list=event_list)

    cmptor = Comparator(event_params_dict=event_params_dict)
    label = cmptor.predict(binary_sequences[0])
    print label
    return label


def trainAll(source_tag, target_tag, obs_len, obs_count, algo_type):
    '''train randomly all
    '''
    # Get events' info from db.
    events = getEventInfo()

    for event in events:
        trainEventRandomly(event, source_tag, target_tag, obs_len, obs_count, algo_type)

    return True


def predictEvent(seq, tag, algo_type, x_request_id=''):
    '''seq最可能属于一个tag下哪个label的model

    Parameters
    ----------
    seq: list
    tag: string
    algo_type: string

    Returns
    -------
    predict_result: dict
      e.g. {"shopping": 0.7, "sleeping": 0.3}
    '''
    algo_type2classifer_map = {"GMMHMM": classifier.GMMHMMClassifier}

    logger.info('<%s>, [predict event] start get Model by tag:%s' % (x_request_id, tag))
    models = {}
    for model in getModelByTag(algo_type, tag):
        models[model.get('eventType')] = {'status_set': model.get('statusSets'), 'param': model.get('param')}
    logger.info('<%s>, [predict event] end get Model by tag:%s' %(x_request_id, tag))

    if not models or len(models) == 0:
        logger.error("<%s>, [predict event] tag=%s don't have models" % (x_request_id, tag))
        raise ValueError("tag=%s don't have models" % (tag))

    logger.info('<%s>, [predict event] start predict, seq=%s' % (x_request_id, seq))
    CLASSIFER = algo_type2classifer_map[algo_type]
    my_classifer = CLASSIFER(models)
    predict_result = my_classifer.predict(seq)
    logger.info('<%s>, [predict event] end predict, seq=%s, predict_result=%s' %(x_request_id, seq, predict_result))

    return predict_result



if __name__ == "__main__":


    pass
