from settings import configList

def checkConfigValidation(config_list_db):
    for config in configList:
        if config in config_list_db:
            continue
        else:
            return False
    return True