import random
import scipy.stats as ss
import sys, traceback

def standardNormalRand(range_x, range_y):
    '''
    Standard Normal Rand

    Generate a standard normal rand number,
    the X axis ranges from -1*range_x to range_x,
    the Y axis ranges from -1*range_y to range_y
    '''
    while True:
        X = random.uniform((-1)*range_x, range_x)
        Y = random.uniform(0.0, range_y)
        if Y < ss.norm.pdf(X):
            return abs(X)

def discreteSpecifiedRand(prob_dict_list):
    '''
    Discrete Specified Rand
    return a key which chosen from prob_dict_list in a specified discrete distribution.
    the input format look like:
        [{key: prob1}, {key: prob2}, ...]

    :param prob_dict_list:
    :return:
    '''
    rand  = random.random()
    # print 'rand is', rand
    scale = prob_dict_list[0].values()[0]
    index = 0
    print('index=%s, len=%s, prob_dict_list=%s' % (index, len(prob_dict_list), prob_dict_list))
    while index <= len(prob_dict_list)-1:
        # print 'scale is', scale
        print('index=%s, rand=%s, scale=%s' % (index, rand, scale))
        if rand <= scale:
            return prob_dict_list[index].keys()[0]
        elif rand > scale:
            index += 1
            scale += prob_dict_list[index].values()[0]

def chooseRandomly(choice_list):
    '''
    Choose Randomly

    Choose a possible selection from choice_list,
    and return it.
    '''
    return random.choice(choice_list)

def selectOtherRandomly(prob_dict_list, universal_set):
    '''
    Select Other Randomly

    Select a selection from Others randomly,
    the Others' set is unversal_set exclude prob
    '''
    _universal_set = list(universal_set)
    # delete the existed keys from universal set
    for item in prob_dict_list:
        if item.keys()[0] != 'Others':
            _universal_set.remove(item.keys()[0])
    # delete the key 'Others', and replace it by a new key which generated randomly
    index = 0
    while index < len(prob_dict_list):
        if prob_dict_list[index].keys()[0] == 'Others':
            prob_dict_list[index][random.choice(_universal_set)] = prob_dict_list[index].pop('Others')
        index += 1
    return prob_dict_list

def getTracebackInfo():
    _, _, exc_traceback = sys.exc_info()
    traceback_details = []
    for filename, linenum, funcname, source in traceback.extract_tb(exc_traceback):
        t_d = "%-23s:%s '%s' in %s()" % (filename, linenum, source, funcname)
        traceback_details.append(t_d)

    return traceback_details
