# coding: utf-8

__all__ = ["getModel", "setModel", "getModelByTag"]

from leancloud import Object
from leancloud import Query

Model = Object.extend("Model")

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
    result = Query.do_cloud_query('select * from Model where algoType="%s" and tag="%s"' % (algo_type, model_tag))
    results = result.results

    # get most recent models
    models_label_set = set()
    for model in results:
        models_label_set.add(model.get('eventType'))

    recent_models_list = []
    for label in models_label_set:
        result = Query.do_cloud_query('select * from Model where algoType="%s" and tag="%s" and eventType="%s" limit 1 order by -updatedAt'
                                      % (algo_type, model_tag, label))
        results = result.results
        recent_models_list.append(results[0])

    return recent_models_list
