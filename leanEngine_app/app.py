# coding: utf-8

import os
import datetime
import time
import json
import logging

from flask import Flask, request, make_response
from logentries import LogentriesHandler
import bugsnag
from bugsnag.flask import handle_exceptions
from leancloud.errors import LeanCloudError

from config import *
from event_analyzer_lib import core, utils


# Configure Logentries
logger = logging.getLogger('logentries')
if APP_ENV == 'prod':
    logger.setLevel(logging.INFO)
else:
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s, %(message)s'))
    logger.addHandler(ch)
logentries_handler = LogentriesHandler(LOGENTRIES_TOKEN)
logger.addHandler(logentries_handler)

# Configure Bugsnag
bugsnag.configure(
    api_key=BUGSNAG_TOKEN,
    project_root=os.path.dirname(os.path.realpath(__file__))
)

app = Flask(__name__)

# Attach Bugsnag to Flask's exception handler
handle_exceptions(app)


@app.before_first_request
def init_before_first_request():
    init_tag = "[Initiation of Service Process]\n"
    logger.info('[init] enter init_before_first_request')

    log_init_time = "Initiation START at: \t%s\n" % (datetime.datetime.now())
    log_app_env = "Environment Variable: \t%s\n" % (APP_ENV)
    log_bugsnag_token = "Bugsnag Service TOKEN: \t%s\n" % (BUGSNAG_TOKEN)
    log_logentries_token = "Logentries Service TOKEN: \t%s\n" % (LOGENTRIES_TOKEN)
    logger.info(init_tag + log_init_time)
    logger.info(init_tag + log_app_env)
    logger.info(init_tag + log_bugsnag_token)
    logger.info(init_tag + log_logentries_token)

@app.route('/init/', methods=['POST'])
def rebuild_event():
    '''init specify event_type & tag model

    Parameters
    ----------
    data: JSON Obj
      e.g. {"event_type":"shopping#mall", "tag":"init_model"}
      event_type: string
      tag: string
      algo_type: string, optional, default "GMMHMM"

    Returns
    -------
    result: JSON Obj
      e.g. {"code":0, "message":"success", "result":""}
      code: int
        0 success, 1 fail
      message: string
      result: dict, optional
        model object id
    '''
    if request.headers.has_key('X-Request-Id') and request.headers['X-Request-Id']:
        x_request_id = request.headers['X-Request-Id']
    else:
        x_request_id = ''

    logger.info('<%s>, [init] enter, request ip:%s, ua:%s' %(x_request_id, request.remote_addr, request.remote_user))
    result = {'code': 1, 'message': ''}

    # params JSON validate
    try:
        incoming_data = json.loads(request.data)
    except ValueError, err_msg:
        logger.exception('<%s>, [init] [ValueError] err_msg: %s, params=%s' % (x_request_id, err_msg, request.data))
        result['message'] = 'Unvalid params: NOT a JSON Object'
        result['code'] = 103
        return make_response(json.dumps(result), 400)

    # params key checking
    for key in ['event_type', 'tag']:
        if key not in incoming_data:
            logger.exception("<%s>, [init] [KeyError] params=%s, should have key: %s" % (x_request_id, incoming_data, key))
            result['message'] = "Params content Error: cant't find key=%s" % (key)
            result['code'] = 103
            return make_response(json.dumps(result), 400)

    event_type = incoming_data['event_type']
    tag = incoming_data['tag']
    algo_type = incoming_data.get('algo_type', 'GMMHMM')

    logger.info('<%s>, [init] valid request params: %s' % (x_request_id, incoming_data))

    model_id = core.rebuildEvent(event_type, algo_type, tag)
    result['code'] = 0
    result['message'] = 'success'
    result['result'] = {'modelObjectId': model_id}
    logger.info('<%s> [init] success' % (x_request_id))

    return json.dumps(result)


@app.route('/initAll/', methods=['POST'])
def init_all():
    '''init model and store in db with input tag

    Parameters
    ----------
    data: JSON Obj
      e.g. {}
      tag: string, optional, default 'init_model_timestamp'
      algo_type: string, optional, default "GMMHMM"

    Returns
    -------
    result: JSON Obj
      e.g. {"code":0, "message":"success", "result":""}
      code: int
        0 success, 1 fail
      message: string
      result:
    '''
    if request.headers.has_key('X-Request-Id') and request.headers['X-Request-Id']:
        x_request_id = request.headers['X-Request-Id']
    else:
        x_request_id = ''

    logger.info('<%s>, [init all] enter, request ip:%s, ua:%s' %(x_request_id, request.remote_addr, request.remote_user))
    result = {'code': 1, 'message': ''}

    # params JSON validate
    try:
        if request.data:
            incoming_data = json.loads(request.data)
        else:
            incoming_data = {}
    except ValueError, err_msg:
        logger.exception('<%s>, [init all] [ValueError] err_msg: %s, params=%s' % (x_request_id, err_msg, request.data))
        result['message'] = 'Unvalid params: NOT a JSON Object'
        result['code'] = 103
        return make_response(json.dumps(result), 400)

    tag = incoming_data.get('tag', 'init_model_%s'%(int(time.time())))
    algo_type = incoming_data.get('algo_type', 'GMMHMM')

    logger.info('<%s>, [init all] valid request params: %s' % (x_request_id, incoming_data))

    core.initAll(tag, algo_type)
    result['message'] = 'success'
    result['code'] = 0
    logger.info('<%s> [init all] success' % (x_request_id))
    return json.dumps(result)


@app.route('/trainRandomly/', methods=['POST'])
def train_randomly():
    '''Randomly train a `sourceTag` model.

    After train, model params will be saved in db and taged `targetTag`

    Parameters
    ----------
    data: JSON Obj
      e.g. {"event_type":"shopping#mall", "sourceTag":"init_model"}
      event_type: string
      sourceTag: string
      targetTag: string, optional, default "random_train"
      algo_type: string, optional, default "GMMHMM"

    Returns
    -------
    result: JSON Obj
      e.g. {"code":0, "message":"success", "result":""}
      code: int
        0 success, 1 fail
      message: string
      result: dict, optional
        model object id
    '''
    if request.headers.has_key('X-Request-Id') and request.headers['X-Request-Id']:
        x_request_id = request.headers['X-Request-Id']
    else:
        x_request_id = ''

    logger.info('<%s>, [train randomly] enter, request ip:%s, ua:%s' %(x_request_id, request.remote_addr, request.remote_user))
    result = {'code': 1, 'message': ''}

    # params JSON validate
    try:
        incoming_data = json.loads(request.data)
    except ValueError, err_msg:
        logger.exception('<%s>, [tran randomly] [ValueError] err_msg: %s, params=%s' % (x_request_id, err_msg, request.data))
        result['message'] = 'Unvalid params: NOT a JSON Object'
        result['code'] = 103
        return make_response(json.dumps(result), 400)

    # params key checking
    for key in ['event_type', 'sourceTag']:
        if key not in incoming_data:
            logger.exception("<%s>, [train randomly] [KeyError] params=%s, should have key: %s" % (x_request_id, incoming_data, key))
            result['message'] = "Params content Error: cant't find key=%s" % (key)
            result['code'] = 103
            return make_response(json.dumps(result), 400)

    event_type = incoming_data['event_type']
    sourceTag = incoming_data['sourceTag']
    targetTag = incoming_data.get('targetTag', 'random_train')
    algo_type = incoming_data.get('algo_type', 'GMMHMM')
    obs_len   = incoming_data.get('obs_len', 10)
    obs_count = incoming_data.get('obs_count', 500)

    logger.info('<%s>, [train randomly] valid request params: %s' % (x_request_id, incoming_data))

    model_id = core.trainEventRandomly(event_type, sourceTag, targetTag, obs_len, obs_count, algo_type)
    result['code'] = 0
    result['message'] = 'success'
    result['result'] = {'modelObjectId': model_id}
    logger.info('<%s> [train randomly] success' % (x_request_id))

    return json.dumps(result)


@app.route('/trainRandomlyAll/', methods=['POST'])
def train_randomly_all():
    '''Randomly train a `sourceTag` model of all event labels.

    After train, model params will be saved in db and taged `targetTag`

    Parameters
    ----------
    data: JSON Obj
      e.g. {"sourceTag":"init_model"}
      sourceTag: string
      targetTag: string, optional, default "random_train"
      algo_type: string, optional, default "GMMHMM"

    Returns
    -------
    result: JSON Obj
      e.g. {"code":0, "message":"success", "result":""}
      code: int
        0 success, 1 fail
      message: string
      result: dict, optional
        model object id
    '''
    if request.headers.has_key('X-Request-Id') and request.headers['X-Request-Id']:
        x_request_id = request.headers['X-Request-Id']
    else:
        x_request_id = ''

    logger.info('<%s>, [train randomly all] enter, request ip=%s, ua=%s' %(x_request_id, request.remote_addr, request.remote_user))
    result = {'code': 1, 'message': ''}

    # params JSON validate
    try:
        incoming_data = json.loads(request.data)
    except ValueError, err_msg:
        logger.exception('<%s>, [tran randomly all] [ValueError] err_msg: %s, params=%s' % (x_request_id, err_msg, request.data))
        result['message'] = 'Unvalid params: NOT a JSON Object'
        return json.dumps(result)

    # params key checking
    for key in ['sourceTag']:
        if key not in incoming_data:
            logger.exception("<%s>, [train randomly all] [KeyError] params=%s, should have key: %s" % (x_request_id, incoming_data, key))
            result['message'] = "Params content Error: cant't find key=%s" % (key)
            return json.dumps(result)

    logger.info('<%s>, [train randomly all] valid request params: %s' % (x_request_id, incoming_data))

    sourceTag = incoming_data['sourceTag']
    targetTag = incoming_data.get('targetTag', 'random_train')
    algo_type = incoming_data.get('algo_type', 'GMMHMM')
    obs_len   = incoming_data.get('obs_len', 10)
    obs_count = incoming_data.get('obs_count', 500)

    core.trainAll(sourceTag, targetTag, obs_len, obs_count, algo_type)
    result['code'] = 0
    result['message'] = 'success'
    logger.info('<%s> [train randomly all] success' % (x_request_id))

    return json.dumps(result)

@app.route('/predict/', methods=['POST'])
def predict():
    '''Predict seq belong to which event

    Parameters
    ----------
    data: JSON obj
      e.g. {
            "seq" : [{"motion": "sitting", "sound": "unknown", "location": "chinese_restaurant"},
                     {"motion": "sitting", "sound": "shop", "location": "chinese_restaurant"},
                     {"motion": "walking", "sound": "shop", "location": "night_club"}],
            "tag":"randomTrain"
           }
      seq: list
      tag: string
      algo_type: string, optional, default "GMMHMM"

    Returns
    -------
    result: JSON Obj
      e.g. {"code":0, "message":"success", "result":{"shopping":0.7,"walking":0.3}}
      code: int
        0 success, 1 fail
      message: string
      result: dict
    '''
    if request.headers.has_key('X-Request-Id') and request.headers['X-Request-Id']:
        x_request_id = request.headers['X-Request-Id']
    else:
        x_request_id = ''

    logger.info('<%s>, [predict] enter, request ip:%s, ua:%s' %(x_request_id, request.remote_addr, request.remote_user))
    result = {'code': 1, 'message': ''}

    # params JSON validate
    try:
        incoming_data = json.loads(request.data)
    except ValueError, err_msg:
        logger.exception('<%s>, [predict] [ValueError] err_msg: %s, params=%s' % (x_request_id, err_msg, request.data))
        result['message'] = 'Unvalid params: NOT a JSON Object'
        result['code'] = 103
        return make_response(json.dumps(result), 400)

    # params key checking
    for key in ['seq', 'tag']:
        if key not in incoming_data:
            logger.exception("<%s>, [predict] [KeyError] params=%s, should have key: %s" % (x_request_id, incoming_data, key))
            result['message'] = "Params content Error: cant't find key=%s" % (key)
            result['code'] = 103
            return make_response(json.dumps(result), 400)

    seq = incoming_data['seq']
    tag = incoming_data['tag']
    algo_type = incoming_data.get('algo_type', "GMMHMM")

    if not seq:
        result['code'] = 103
        result['message'] = 'input params [seq=NULL]'
        logger.info('<%s>, [predct] request params `seq`=NULL' % (x_request_id))
        return json.dumps(result)

    logger.info('<%s>, [predict] valid request params seq=%s, tag=%s, algo_type=%s' % (x_request_id, seq, tag, algo_type))

    try:
        # data clean for seq
        seq_cleaned = []
        for elem in seq:
            seq_cleaned.append({
                'location': elem['location'],
                'motion': elem['motion'],
                'sound': elem['sound'],
            })

        result['result'] = core.predictEvent(seq_cleaned, tag, algo_type, x_request_id)
        result['code'] = 0
        result['message'] = 'success'
        logger.info('<%s> [predict] success, predict result=%s' %(x_request_id, result['result']))
    except LeanCloudError, err_msg:
        result['message'] = "[LeanCloudError] Maybe can't find tag=%s" % (tag)
        logger.info('<%s> [predict] [LeanCloudError] %s' % (x_request_id, err_msg))
        result['code'] = 103
        return make_response(json.dumps(result), 400)

    return json.dumps(result)


@app.route('/train/', methods=['POST'])
def train():
    '''Train a `sourceTag` model of `event_type` by `obs`.

    After train, model params will be saved in db and taged `targetTag`

    Parameters
    ----------
    data: JSON Obj
      e.g. {
            "obs" : [{"motion": "sitting", "sound": "unknown", "location": "chinese_restaurant"},
                     {"motion": "sitting", "sound": "shop", "location": "chinese_restaurant"},
                     {"motion": "walking", "sound": "shop", "location": "night_club"}],
            "sourceTag": "init_model",
            "event_type": "dining_in_restaurant"
           }
      obs: list, must be 2-dimension list
      event_type: string
      sourceTag: string
      targetTag: string, optional, default equal to `sourceTag`
      algo_type: string, optional, default "GMMHMM"

    Returns
    -------
    result: JSON Obj
      e.g. {"code":0, "message":"success", "result":""}
      code: int
        0 success, 1 fail
      message: string
      result: dict, optional
        model object id
    '''
    if request.headers.has_key('X-Request-Id') and request.headers['X-Request-Id']:
        x_request_id = request.headers['X-Request-Id']
    else:
        x_request_id = ''

    logger.info('<%s>, [train] enter, request ip:%s, ua:%s' %(x_request_id, request.remote_addr, request.remote_user))
    result = {'code': 1, 'message': ''}

    # params JSON validate
    try:
        incoming_data = json.loads(request.data)
    except ValueError, err_msg:
        logger.exception('<%s>, [train] [ValueError] err_msg: %s, params=%s' % (x_request_id, err_msg, request.data))
        result['message'] = 'Unvalid params: NOT a JSON Object'
        result['code'] = 103
        return make_response(json.dumps(result), 400)

    # params key checking
    for key in ['obs', 'event_type', 'sourceTag']:
        if key not in incoming_data:
            logger.exception("<%s>, [train] [KeyError] params=%s, should have key: %s" % (x_request_id, incoming_data, key))
            result['message'] = "Params content Error: cant't find key=%s" % (key)
            result['code'] = 103
            return make_response(json.dumps(result), 400)

    event_type = incoming_data['event_type']
    sourceTag = incoming_data['sourceTag']
    obs = incoming_data['obs']
    targetTag = incoming_data.get('targetTag', sourceTag)
    algo_type = incoming_data.get('algo_type', 'GMMHMM')

    if not utils.check_2D_list(obs):
        result['message'] = 'input obs=%s is not 2-dimension' % (obs)
        return json.dumps(result)

    logger.info('<%s>, [train] valid request params: %s' % (x_request_id, incoming_data))

    # TODO: wrapper for obs
    cleaned_obs = []
    for seq in obs:
        seq_cleaned = []
        for elem in seq:
            seq_cleaned.append({
                'location': elem['location'],
                'motion': elem['motion'],
                'sound': elem['sound'],
            })
        cleaned_obs.append(seq_cleaned)

    try:
        model_id = core.trainEvent(obs, event_type, sourceTag, targetTag, algo_type)
        result['code'] = 0
        result['message'] = 'success'
        result['result'] = {'modelObjectId': model_id}
        logger.info('<%s> [train] success' % (x_request_id))
    except LeanCloudError, err_msg:
        result['message'] = "[LeanCloudError] Maybe can't find sourceTag=%s, event_type=%s" % (sourceTag, event_type)
        logger.info('<%s> [train] [LeanCloudError] %s' % (x_request_id, err_msg))
        result['code'] = 103
        return make_response(json.dumps(result), 400)

    return json.dumps(result)


@app.route('/isAlive/', methods=['GET'])
def is_alive():
    '''API for server alive test
    '''
    if request.headers.has_key('X-Request-Id') and request.headers['X-Request-Id']:
        x_request_id = request.headers['X-Request-Id']
    else:
        x_request_id = ''

    logger.info('<%s>, [isAlive] request from ip:%s, ua:%s' %(x_request_id, request.remote_addr,
                                                                       request.remote_user))
    result = {'code': 0, 'message': 'Alive'}
    return json.dumps(result)



