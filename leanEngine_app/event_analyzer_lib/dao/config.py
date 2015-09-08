__all__ = ["getSystemStatusSets", "getEventInfo", "getEventList", "getEventProbMap","get_location_one_set","get_location_two_set","get_motion_set","get_sound_set"]

from leancloud import Object
from leancloud import Query
from settings import configList

Config = Object.extend("Config")

# Basic Function

def _getConfig(config_name):
    query = Query(Config)
    query.equal_to("name", config_name)
    config_value = query.first().get("value")
    return config_value

def _getConfigList():
    query = Query(Config)
    config_list = []
    for config in query.find():
        config_list.append(config.get("name"))
    return config_list


# Advanced Function


def get_location_one_set():

    location_one_dict = _getConfig(configList[2])
    #todo checck isActive
    return location_one_dict.keys()

def get_location_two_set():

    #todo checck isActive
    location_two_dict = _getConfig(configList[3])
    return location_two_dict.keys()

def get_motion_set():
    #todo checck isActive
    motion_dict = _getConfig(configList[4])
    return motion_dict.keys()

def get_sound_set():
    print configList[5]
    sound_dict = _getConfig(configList[5])
    return sound_dict.keys()


def getSystemStatusSets():
    status = _getConfig(configList[0])
    # Get Sets of all kinds of status classification.
    sys_status_sets = {}
    for status_type in status:
        status_config_name       = status[status_type]["selected"]
        status_config_value      = _getConfig(status_config_name)
        status_classification    = []
        for classification_item in status_config_value:
            if status_config_value[classification_item]["isActive"] is True:
                status_classification.append(classification_item)
        sys_status_sets[status_type] = status_classification
    return sys_status_sets

def getEventInfo():
    # Get all events' info from db.
    events = _getConfig(configList[1])
    # Get rid of inactive event.
    for event in events:
        if events[event]["isActive"] is False:
            events.pop(event)
    return events

def getEventList():
    event_list = []
    # Get all events' info from db.
    events = _getConfig(configList[1])
    # Get rid of inactive event.
    for event in events:
        if events[event]["isActive"] is True:
            event_list.append(event)
    return event_list

def getEventProbMap():
    event_list = getEventList()
    event_prob_map = {}
    #print('!!!!!!! event_prob_map:%s' % (_getConfig('event_prob_map')))
    #print('event_list: %s' % (event_list))
    for key, value in _getConfig('event_prob_map').iteritems():
        if key in event_list:
            event_prob_map[key] = value
    #print('!!!!!!! result:%s' % (event_prob_map))
    print "event_prob_map", event_prob_map

    check_valid_event_prob_map(event_prob_map)
    return event_prob_map

def check_valid_event_prob_map(event_prob_map):
    '''Check event_prob_map valid

    If unvalid, will raise ValueError
    '''
    # for event, cur_event_prob_map in event_prob_map.iteritems():
    #     for key in ['motion', 'sound', 'location']:
    #         cur_event_prob_dict = {}
    #         print "cur_event_prob_map", cur_event_prob_map
    #         for tmp_key,tmp_value in cur_event_prob_map[key].iteritems():
    #             cur_event_prob_dict.update({tmp_key:tmp_value})
    #         #print(cur_event_prob_dict)
    #         print "values are",cur_event_prob_dict.values()
    #         if 1.0 != round(reduce(lambda x, y: x+y, cur_event_prob_dict.values())):
    #             raise ValueError('event<%s> prob_map[%s] total_probs != 1\n details=%s'
    #                              % (event, key, cur_event_prob_map[key]))

    return True
