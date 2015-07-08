# -*- coding: utf-8 -*-

from dao.config import *
from dao.model import *
from algo.datasets import Dataset
import datetime
import logging
from algo import trainer, classifier

logger = logging.getLogger('logentries')


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


def trainEventRandomly(
        event_type,
        tag,
        algo_type="GMMHMM"
):
    algo_type2trainer_map = {'GMMHMM': trainer.GMMHMMTrainer}

    model = getModel(algo_type, tag, event_type)
    model_param = model["modelParam"]
    status_sets = model["statusSets"]

    sound_set = status_sets["sound"]
    motion_set = status_sets["motion"]
    location_set = status_sets["location"]

    train_obs_len = 10
    train_obs_count = 30
    d = Dataset(event_type=getEventList(), motion_type=motion_set, sound_type=sound_set,
                location_type=location_set, event_prob_map=getEventProbMap())
    d.randomObservations(event_type, train_obs_len, train_obs_count)
    observations = d.obs
    logger.debug('[trainEventRandomly] %s' % (d))

    description = 'Random train algo_type=%s for eventType=%s, random train obs_len=%s, obs_count=%s' % (
        algo_type, event_type, train_obs_len, train_obs_count)
    TRAINER = algo_type2trainer_map[algo_type]
    my_trainer = TRAINER(model_param)
    my_trainer.fit(d.getDataset())

    return setModel(algo_type, 'random_train', event_type, my_trainer.params_, status_sets,
                    datetime.datetime.now(), description, observations)


def predictEvent(seq, tag, algo_type):
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

    models = {}
    for model in getModelByTag(algo_type, tag):
        models[model.get('eventType')] = {'status_set': model.get('statusSets'), 'param': model.get('param')}

    if not models or len(models) == 0:
        raise ValueError("tag=%s don't have models" % (tag))

    CLASSIFER = algo_type2classifer_map[algo_type]
    my_classifer = CLASSIFER(models)
    logger.info('[predict event] model load success')

    return my_classifer.predict(seq)
