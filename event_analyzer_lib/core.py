from dao.config import *
from dao.model import *
from dataset import Dataset
import datetime

def rebuildEvent(
        event_type,
        algo_type="GMMHMM",
        new_tag="TAG_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
):
    # Get events' info from db.
    events = getEventInfo()
    # Validation of existance of the event
    if event_type not in events:
        print "There is no event named %s in Config Class" % event_type
        return None

    # Get Model's init params.
    init_params = events[event_type]["initParams"][algo_type]
    # # Get Sets of all kinds of status classification.
    sys_status_sets = getSystemStatusSets()

    print "Event %s's params is %s" % (event_type, init_params)
    print "And latest status classification are %s" % sys_status_sets
    print "The generated model tag is %s" % new_tag

    now = datetime.datetime.now()
    description = "Initiation of A new %s Model for event %s was made at %s" % (algo_type, event_type, now)
    return setModel(algo_type=algo_type, model_tag=new_tag, event_type=event_type,
                  model_param=init_params, status_sets=sys_status_sets, timestamp=now, description=description)

def trainEventRandomly(
        event_type,
        tag,
        algo_type="GMMHMM"
):
    model = getModel(algo_type, tag, event_type)
    model_param = model["modelParam"]
    status_sets = model["statusSets"]

    sound_set    = status_sets["sound"]
    motion_set   = status_sets["motion"]
    location_set = status_sets["location"]

    d = Dataset(event_type=getEventList(), motion_type=motion_set, sound_type=sound_set,
                location_type=location_set, event_prob_map={})
    

def predictEvent():
    pass