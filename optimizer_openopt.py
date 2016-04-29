#!/usr/bin/python
'''
Simplest OpenOpt KSP example;
requires FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/KSP for more details
'''

import pandas as pd

from openopt import *
import math
import csv
from pprint import pprint
import datetime
from proj_elastic import InsertProj
from fuzzywuzzy import process
import numpy as np


def load_projections(projections_file):
    projections = {}
    with open(projections_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            projections[row[0]] = float(row[1])
    return projections

''' fuzzy match
    inputs - list of strings typed by user
           - list of options for players
    outputs - list of fuzzy matched strings
'''


def fuzzy_match(inputlist, choices):
    if (isinstance(inputlist, str)):
        fz = process.extractOne(inputlist, choices)
        return(fz[0])
    else:
        outlist = []
        for s in inputlist:
            fz = process.extractOne(s, choices)
            outlist.append(fz[0])
        return(outlist)


def optimize_grid(delta=[-4, -2, 0, 2, 4], min_dvp=[0, 1, 2, 3], min_sal=[2000, 2500,
                                                                          3000], exclusions=[], locks=[], target=['DK_Proj', 'min_proj', 'proj_pure']):

    df = pd.DataFrame()
    for d in delta:
        for v in min_dvp:
            for m in min_sal:
                for t in target:
                    players = optimizer(
                        delta=d,
                        min_dvp=v,
                        min_sal=m,
                        exclusions=exclusions,
                        locks=locks,
                        target=t,
                        return_format='df')
                    df = df.append(players)
    return(df)


def optimizer(locks=[], exclusions=[], delta=0, min_own=0, min_dvp=0,
              min_sal=2000, max_own=100, target='DK_Proj', return_format='ajax'):

    adjustments = {}

    items = []
    player_ids = {}
    today = datetime.datetime.today() - datetime.timedelta(hours=4)
    yesterday = today - datetime.timedelta(hours=24)
    path = '/home/ubuntu/dfsharp/opt_csvs/' + \
        today.strftime('%Y%m%d') + '_opt.csv'
    ypath = '/home/ubuntu/dfsharp/opt_csvs/' + \
        yesterday.strftime('%Y%m%d') + '_opt.csv'
    abblist = ['bkn', 'bos', 'cha', 'chi', 'cle', 'dal', 'den', 'det', 'hou',
               'ind', 'lac', 'lal', 'mem', 'mia', 'mil', 'min', 'nor', 'nyk',
               'okc', 'orl', 'phi', 'pho', 'por', 'sac', 'sas', 'tor', 'uta',
               'was', 'atl', 'gsw']

    # read csv of players
    try:
        df = pd.read_csv(path)
    except IOError:
        df = pd.read_csv(ypath)
        path = ypath

    playernames = df['name'].tolist()
    teamnames = list(df['Team'].unique())
    if len(locks) > 0:
        locks = fuzzy_match(locks, playernames)
    # loop through exclusions, fuzzy teams one way and players another
    if len(exclusions) > 0:
        combo = teamnames + playernames
        exclusions = fuzzy_match(exclusions, combo)

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        index = 0
        for row in reader:
            if (row[1] in locks) or ((index != 0) and (row[1] not in exclusions) and (row[11] not in exclusions) and (float(row[13]) >= float(min_own)) and (
                    float(row[13]) <= float(max_own)) and (float(row[16]) >= float(min_dvp)) and (float(row[4]) > 5) and (float(row[2]) > float(min_sal))):
                vals = {
                    'id': index - 1,
                    'PG': 1 if row[0] == 'PG' else 0,
                    'SG': 1 if row[0] == 'SG' else 0,
                    'SF': 1 if row[0] == 'SF' else 0,
                    'PF': 1 if row[0] == 'PF' else 0,
                    'C': 1 if row[0] == 'C' else 0,
                    'name': row[1],
                    'salary': int(row[2]),
                    'DK_Proj': float(row[4]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    'min_proj': float(row[8]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    'dk_per_min': float(row[9]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    'lock': 1 if row[1] in locks else 0,
                    #'ownership': float(row[13]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    'dvprank': float(row[16]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    #'otprank': float(row[17]) if row[1] not in adjustments.keys() else adjustments[row[1]]
                    'usage_5g_avg': float(row[19]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    'value_3g_avg': float(row[28]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    'proj_pure': float(row[31]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                    'dk_std_90_days': float(row[33]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                }
                for team in abblist:
                    vals[team] = 1 if row[11] == team else 0
                vals['PGSGC'] = vals['PG'] + vals['SG'] + vals['C']
                vals['PFSFC'] = vals['PF'] + vals['SF'] + vals['C']
                # if projections != None:
                #    vals['fpts'] = projections[vals['name']]
                items.append(vals)
            index += 1

    for item in items:
        for i in range(len(items)):
            item['id%d' % i] = float(item['id'] == i)

    constraints = lambda values: (
        values['lock'] == len(locks),
        values['salary'] >= 49500,
        #values['salary'] != 50100,
        values['salary'] <= int(50000),
        values['nItems'] == 8,
        values['PG'] >= 1,
        values['PG'] <= 2,
        values['SG'] >= 1,
        values['SG'] <= 2,
        values['SF'] >= 1,
        values['SF'] <= 2,
        values['PF'] >= 1,
        values['PF'] <= 2,
        values['C'] >= 1,
        values['PFSFC'] >= 4,
        values['PFSFC'] <= 5,
        values['PGSGC'] >= 4,
        values['PGSGC'] <= 5,
    ) + tuple([values['id%d' % i] <= 1 for i in range(len(items))])

    #objective = lambda val: val['fpts'] + delta*val['otprank']
    objective = lambda val: val[target] + delta * val['dk_std_90_days']
    p = KSP(objective, items, goal='max', constraints=constraints)
    # requires cvxopt and glpk installed, see http://openopt.org/KSP for other
    # solvers
    r = p.solve('glpk', iprint=0)
    ''' Results for Intel Atom 1.6 GHz:
    ------------------------- OpenOpt 0.50 -------------------------
    solver: glpk   problem: unnamed    type: MILP   goal: max
     iter   objFunVal   log10(maxResidual)
        0  0.000e+00               0.70
        1  2.739e+01            -100.00
    istop: 1000 (optimal)
    Solver:   Time Elapsed = 0.82   CPU Time Elapsed = 0.82
    objFunValue: 27.389749 (feasible, MaxResidual = 0)
    '''
    # print(r.xf)
    playerlist = r.xf

    # r.xf is a list of players- we will merge their info back and return a DF
    # instead
    df2 = df[df['name'].isin(playerlist)]
    # df2 is the latest lineup - we'll return the frame [for now]
    df2[['DK_Proj', 'min_proj', 'dk_per_min', 'value', 'usage_5g_avg']] = np.round(
        df2[['DK_Proj', 'min_proj', 'dk_per_min', 'value', 'usage_5g_avg']], 1)
    ajax = df2[['numpos',
                'name',
                'Team',
                'Opp',
                'dk_sal',
                'ownership',
                'DK_Proj',
                'dvprank',
                'min_proj',
                'dk_per_min',
                'value',
                'usage_5g_avg']].to_json(orient='records')
    #ajax = df2.to_html(index=False)

    InsertProj(df2, indexer="latestlineup")
    # return(playerlist)
    print(constraints)
    if(return_format == 'ajax'):
        return(ajax)
    else:
        return(df2)
