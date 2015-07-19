__author__ = 'jiaying.lu'

__all__ = ['APP_ID', 'MASTER_KEY', 'APP_ENV', 'LOGENTRIES_TOKEN', 'BUGSNAG_TOKEN']

import os

# for LeanEngine
APP_ID = ''
MASTER_KEY = ''

LOCAL_APP_ID = '4gm6cm40flbj9url010qom89srqwgphpf77643cmjkwng10u'
TEST_APP_ID = '4gm6cm40flbj9url010qom89srqwgphpf77643cmjkwng10u'
PROD_APP_ID = 'ea930cifzb6y8sfsvbud8m9yng0utfyy49rhfxr1yvm0y3il'

LOCAL_MASTER_KEY = '1cjz8wpxasw1i150cx25foyy0e333scqx6rs5u9v4trfn2fc'
TEST_MASTER_KEY = '1cjz8wpxasw1i150cx25foyy0e333scqx6rs5u9v4trfn2fc'
PROD_MASTER_KEY = '9rh5cpm89l5msjo0m3k0qn30e4ulj79vgw1wdhieajwpuf4o'

# for log & exception
LOGENTRIES_TOKEN = ''
BUGSNAG_TOKEN = ''

LOCAL_LOGENTRIES_TOKEN = 'ea12025e-dcb9-4938-9785-e752d4dca821'
TEST_LOGENTRIES_TOKEN = 'ea12025e-dcb9-4938-9785-e752d4dca821'
PROD_LOGENTRIES_TOKEN = '14f4a1e5-57e6-4e4a-9be7-d2c59afa7232'

LOCAL_BUGSNAG_TOKEN = '824ab4c48cc1ef6a318cf9594bc70226'
TEST_BUGSNAG_TOKEN = '824ab4c48cc1ef6a318cf9594bc70226'
PROD_BUGSNAG_TOKEN = '53fc7707fc1492d15261bed37a5841df'

# Choose keys according to APP_ENV
if os.environ.get('LC_APP_PROD') == '1':
    # prod environ
    APP_ENV = 'prod'
elif os.environ.get('LC_APP_PROD') == '0':
    # test environ
    APP_ENV = 'test'
else:
    # dev environ
    APP_ENV = 'local'
print('-'*20 + '\n|  APP_ENV = %s |\n' %(APP_ENV) + '-'*20)
if APP_ENV == 'test':
    APP_ID = TEST_APP_ID
    MASTER_KEY = TEST_MASTER_KEY
    LOGENTRIES_TOKEN = TEST_LOGENTRIES_TOKEN
    BUGSNAG_TOKEN = TEST_BUGSNAG_TOKEN
elif APP_ENV == 'prod':
    APP_ID = PROD_APP_ID
    MASTER_KEY = PROD_MASTER_KEY
    LOGENTRIES_TOKEN = PROD_LOGENTRIES_TOKEN
    BUGSNAG_TOKEN = PROD_BUGSNAG_TOKEN
elif APP_ENV == 'local':
    APP_ID = TEST_APP_ID
    MASTER_KEY = TEST_MASTER_KEY
    LOGENTRIES_TOKEN = LOCAL_LOGENTRIES_TOKEN
    BUGSNAG_TOKEN = LOCAL_BUGSNAG_TOKEN
else:
    raise ValueError('Unvalid APP_ENV: %s' %(APP_ENV))
