from leancloud import Object
from leancloud import Query

Behavior = Object.extend("Behavior")

def saveUserBehavior(behavior_sequence, source, event_type, model_id, timestamp):
    behavior = Behavior()
    behavior.set("behaviorData", behavior_sequence)
    behavior.set("source", source)
    behavior.set("modelId", model_id)
    behavior.set("eventType", event_type)
    behavior.set("happenedAt", timestamp)
    behavior.save()
    return behavior.id