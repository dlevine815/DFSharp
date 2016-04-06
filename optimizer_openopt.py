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
#from datetime import datetime, timedelta


import objgraph
import gc


def load_projections(projections_file):
    projections = {}
    with open(projections_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            projections[row[0]] = float(row[1])
    return projections
        

def optimizer(locks=[], exclusions=[], delta=0):

    adjustments={}

    items = []
    player_ids = {}
    today = datetime.datetime.today() - datetime.timedelta(hours=4)
    path = '/home/ubuntu/dfsharp/opt_csvs/'+today.strftime('%Y%m%d')+'_opt.csv'
    #path = '/home/ubuntu/dfsharp/opt_csvs/20160329_opt.csv'

    abblist = ['bkn', 'bos', 'cha', 'chi', 'cle', 'dal', 'den', 'det', 'hou',
       'ind', 'lac', 'lal', 'mem', 'mia', 'mil', 'min', 'nor', 'nyk',
       'okc', 'orl', 'phi', 'pho', 'por', 'sac', 'sas', 'tor', 'uta',
       'was', 'atl', 'gsw']

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        index = 0
        for row in reader:
            if (index != 0) and (row[1] not in exclusions) and (row[11] not in exclusions):
                vals = { 
                        'id': index-1,
                        'PG': 1 if row[0] == 'PG' else 0,
                        'SG': 1 if row[0] == 'SG' else 0,
                        'SF': 1 if row[0] == 'SF' else 0,
                        'PF': 1 if row[0] == 'PF' else 0,
                        'C': 1 if row[0] == 'C' else 0,
                        'name': row[1],
                        'salary': float(row[2]),
                        'fpts': float(row[4]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                        'min_proj': float(row[8]) if row[1] not in adjustments.keys() else adjustments[row[1]],
                        'dk_per_min': float(row[9]) if row[1] not in adjustments.keys() else adjustments[row[1]],
			'lock': 1 if row[1] in locks else 0,
			'ownership': float(row[13]) if row[1] not in adjustments.keys() else adjustments[row[1]]
                        }
		for team in abblist:
		    vals[team] = 1 if row[11] == team else 0
                vals['PGSGC'] = vals['PG'] + vals['SG'] + vals['C']
                vals['PFSFC'] = vals['PF'] + vals['SF'] + vals['C']
                #if projections != None:
                #    vals['fpts'] = projections[vals['name']]
                items.append(vals)
            index += 1

    for item in items:
        for i in range(len(items)):
            item['id%d' % i] = float(item['id'] == i)


 

    constraints = lambda values: (
			      values['lock'] == len(locks),
                              values['salary'] < 50000, 
			      values['salary'] > 49500,
                              values['nItems'] == 8, 
			      values['ind'] <= 2,
			      #values['gsw'] <= 2,
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
                             ) + tuple([values['id%d'% i] <= 1 for i in range(len(items))])
			       #+ tuple([values['id%d'% i] == 1 for i in range(len(locks))]) 


                                  # we could use lambda-func, e,g.
                                  # values['mass'] + 4*values['volume'] < 100
    #objective = 'ownership'
    # we could use lambda-func, e.g. 
    #objective = lambda val: .01*val['min_proj']+val['fpts']
    #objective = lambda val: -.5*val['ownership']+val['fpts']
    objective = lambda val: val['fpts']+ delta*val['ownership']
    # objective = lambda val: 5*value['cost'] - 2*value['volume'] - 5*value['mass'] + 3*val['nItems']
    p = KSP(objective, items, goal = 'max', constraints = constraints) 
    r = p.solve('glpk', iprint = 0) # requires cvxopt and glpk installed, see http://openopt.org/KSP for other solvers
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
    print(r.xf) 
    playerlist = r.xf

    # r.xf is a list of players- we will merge their info back and return a DF instead
    df = pd.read_csv(path)
    df2 = df[df['name'].isin(playerlist)]
    #print(df2)

    InsertProj(df2, indexer="latestlineup")

    #gc.collect()
    #print(objgraph.show_most_common_types())
    
    return(playerlist)

#optimizer(locks=['Kobe Bryant','Avery Bray'])

