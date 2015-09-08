# coding: utf-8

__all__ = ["getModel", "setModel", "getModelByTag", "save_rnnrbm_params", "get_all_rnnrbm_params"]

from leancloud import Object
from leancloud import Query
import gevent
import logging

logger = logging.getLogger('logentries')
Model = Object.extend("Model")
Rnnrbm = Object.extend("Rnnrbm")
# Store models in memory
Model_In_Memory = {}

def getModel(algo_type, model_tag, event_type):
    query = Query(Model)
    query.equal_to("algoType", algo_type)
    query.equal_to("tag", model_tag)
    query.equal_to("eventType", event_type)
    query.descending("timestamp")
    model_info  = query.first()
    model_param = model_info.get("param")
    status_sets = model_info.get("statusSets")
    return {
        "modelParam": model_param,
        "statusSets": status_sets
    }

def save_rnnrbm_params(params_dict, base_set_dict, tag="random_v0" ):
    '''

    :param params_dict:
    :param base_set_dict:
    :param tag:
    :return:
    '''
    #try:

    def ndarray_to_list_in_dict(params_dict):

        new_params_dict ={}
        for key, params in params_dict.items():
            new_params_dict.update({key:params.tolist()})
        return new_params_dict


    from datetime import datetime
    trainedAt = datetime.now()
    ids = []
    for event, model_params in params_dict.items():
        data_dict = dict(trainedAt=trainedAt,
                     tag=tag,
                     eventType=event,
                     baseSetDict=base_set_dict,
                     params=ndarray_to_list_in_dict(model_params))
        obj_id = _set_rnnrbm_params(data_dict)
        ids.append(obj_id)
    return ids

    #except Exception,e:
        #print "Exception is",e

    # def ndarray_to_list(params_dict):
    #
    #     list_dict = {}
    #     for event,params in params_dict.items():
    #         list_dict.update({event:params.tolist()})
    #     return list_dict
        #return e



def _set_rnnrbm_params(params):

    '''

    :param params:
     {"eventType":"dining_in_restaurant","tag":"latest","trainedAt":datetime.datetime.now(),"params":{"W":[],...}}
    :return:
    '''
    model = Rnnrbm()
    model.set(params)
    model.save()
    return model.id

def get_all_rnnrbm_params(tag=None, event_list=None):

    event_params_dict = {}
    for event in event_list:
        params_dict = _get_rnnrbm_params(eventType=event)
        event_params_dict.update({event:params_dict["params"]})

    return event_params_dict


def _get_rnnrbm_params(**conditions):
    '''

    :param conditions:
         {"eventType":"dining_in_restaurant","tag":"latest"}
    :return:

    '''
    query = Query(Rnnrbm)
    query.equal_to("eventType",conditions["eventType"])
    #query.equal_to("note",conditions["note"])
    #query.equal_to("tag","latest")
    query.descending("trainedAt")
    rnnrbm = query.first()
    print "rnnrbm",rnnrbm
    return dict(params=rnnrbm.get("params"))


def setModel(algo_type, model_tag, event_type, model_param, status_sets, timestamp, description, last_train_data=''):
    model = Model()
    model.set("algoType", algo_type)
    model.set("tag", model_tag)
    model.set("eventType", event_type)
    model.set("param", model_param)
    model.set("statusSets", status_sets)
    model.set("trainedAt", timestamp)
    model.set("description", description)
    model.set('lastTrainData', last_train_data)
    model.save()
    return model.id

def getModelByTag(algo_type, model_tag):
    '''
    返回指定 algo_type 和 model_tag 的 Model

    Parameters
    ----------
    algo_type: string
      model's name
    tag: string
      model's tag

    Returns
    -------
    recent_models_list: list
      list of model objs
    '''
    # Try to get models from memory
    # If not, request database
    if algo_type not in Model_In_Memory:
        Model_In_Memory[algo_type] = {}

    if model_tag not in Model_In_Memory[algo_type]:
        Model_In_Memory[algo_type][model_tag] = _getModelByTag_from_db(algo_type, model_tag)

    return Model_In_Memory[algo_type][model_tag]


def _getModelByTag_from_db(algo_type, model_tag):
    '''
    根据tag挑出model，如果tag下的eventType有重复的，选择最新的model.

    Parameters
    ----------
    algo_type: string
      model's name
    tag: string
      model's tag

    Returns
    -------
    recent_models_list: list
      list of model objs
    '''
    logger.debug('[_getModelByTag_from_db] algo_type=%s, model_tag=%s MODELS not in Memory' % (algo_type, model_tag))
    result = Query.do_cloud_query('select * from Model where algoType="%s" and tag="%s"' % (algo_type, model_tag))
    results = result.results

    # get most recent models
    models_label_set = set()
    for model in results:
        models_label_set.add(model.get('eventType'))

    threads = []
    for label in models_label_set:
        # gevent for asyc request
        threads.append(gevent.spawn(async_get_model_by_tag_once, algo_type, model_tag, label))
    gevent.joinall(threads)

    recent_models_list = [thread.value for thread in threads]

    return recent_models_list


def async_get_model_by_tag_once(algo_type, model_tag, label):
    '''wrapper func for gevent
    '''
    result = Query.do_cloud_query('select * from Model where algoType="%s" and tag="%s" and eventType="%s" limit 1 order by -updatedAt'
                                  % (algo_type, model_tag, label))
    results = result.results
    return results[0]

