from leancloud import Object
from leancloud import Query

Config = Object.extend('Config')

def getConfig(config_name):
    query = Query(Config)
    query.equal_to('name', config_name)
    config_value = query.first()
    print config_value
    return config_value