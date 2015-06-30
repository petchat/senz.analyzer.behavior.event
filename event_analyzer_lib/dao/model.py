__all__ = ["getModel", "setModel"]

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

def setModel(algo_type, model_tag, event_type, model_param, status_sets, timestamp, description):
    model = Model()
    model.set("algoType", algo_type)
    model.set("tag", model_tag)
    model.set("eventType", event_type)
    model.set("param", model_param)
    model.set("statusSets", status_sets)
    model.set("trainedAt", timestamp)
    model.set("description", description)
    model.save()
    return model.id