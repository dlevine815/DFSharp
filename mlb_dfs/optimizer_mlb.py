#!/usr/bin/python
import pandas as pd
from openopt import *
import math
import csv
from pprint import pprint
import datetime
import numpy as np
sys.path.append('/home/ubuntu/dfsharp/')
from optimizer_openopt import fuzzy_match

def mlb_optimize(locks=[], exclusions=[], delta=0, maxorder=9,
              min_sal=2000, target='DK_Aggregate', return_format='ajax'):

    items = []

    today = datetime.datetime.today() - datetime.timedelta(hours=4)
    yesterday = today - datetime.timedelta(hours=24)

    abblist = ['SEA', 'SDP', 'HOU', 'ARI', 'TEX', 'CLE', 'WAS', 'PHI', 'BOS',
	       'BAL', 'TOR', 'NYY', 'NYM', 'CHW', 'SFG', 'ATL', 'PIT', 'MIA',
	       'LAD', 'CHC', 'STL', 'MIL', 'TBR', 'KCR', 'COL', 'CIN', 'OAK',
	       'MIN', 'LAA', 'DET']
    try:
        hit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + today.strftime('%Y%m%d') + '_hitters.csv')
	pit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + today.strftime('%Y%m%d') + '_pitchers.csv')
    except IOError:
	hit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + yesterday.strftime('%Y%m%d') + '_hitters.csv')
	pit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + yesterday.strftime('%Y%m%d') + '_pitchers.csv')

    # concat hitters and pitchers to one frame
    cols = ['player','team','dk_sal','dk_pos','DK_Aggregate','DK_Std','matchup','cruz','Cruz_Value',
	    'DK_Rotogrind','DK_Sabersim','DK_Swish','DK_Numberfire','DK_Cafe','DK_Value']
    df = hit[cols]
    df = df.append(pit[cols])

    # populate locks and exclusions 
    playernames = df['player'].tolist()
    teamnames = list(df['team'].unique())
    if len(locks) > 0:
        locks = fuzzy_match(locks, playernames)
    # loop through exclusions, fuzzy teams one way and players another
    if len(exclusions) > 0:
        combo = teamnames + playernames
        exclusions = fuzzy_match(exclusions, combo)

    for index, row in df.iterrows():
        if (row['player'] in locks) or ((row['player'] not in exclusions) and (row['team'] not in exclusions) and (float(row['dk_sal']) > float(min_sal))):
            vals = {
                    'id': index,
                     'SP': 1 if row['dk_pos'] == 'SP' else 0,
                     'C': 1 if row['dk_pos'] == 'C' else 0,
                     '1B': 1 if row['dk_pos'] == '1B' else 0,
                     '2B': 1 if row['dk_pos'] == '2B' else 0,
                     '3B': 1 if row['dk_pos'] == '3B' else 0,
                     'SS': 1 if row['dk_pos'] == 'SS' else 0,
                     'OF': 1 if row['dk_pos'] == 'OF' else 0,
		    #'SP': 1 if 'SP' in row['dk_pos'] == True else 0,
		    #'C': 1 if 'C' in row['dk_pos'] == True else 0,
		    #'1B': 1 if '1B' in row['dk_pos'] == True else 0,
		    #'2B': 1 if '2B' in row['dk_pos'] == True else 0,
		    #'3B': 1 if '3B' in row['dk_pos'] == True else 0,
		    #'SS': 1 if 'SS' in row['dk_pos'] == True else 0,
		    #'OF': 1 if 'OF' in row['dk_pos'] == True else 0,
                    'name': row['player'],
                    'salary': int(row['dk_sal']),
                    'DK_Aggregate': float(row['DK_Aggregate']),
                    'cruz': float(row['cruz']),
                    'DK_Std': float(row['DK_Std']),
                    'lock': 1 if row['player'] in locks else 0,
            }
            for team in abblist:
                vals[team] = 1 if row['team'] == team else 0
            items.append(vals)

    for item in items:
        for i in range(len(items)):
            item['id%d' % i] = float(item['id'] == i)

    constraints = lambda values: (
        values['lock'] == len(locks),
        values['salary'] >= int(49500),
        values['salary'] <= int(50000),
        values['nItems'] == 10,
        values['SP'] == 2,
        values['C'] == 1,
        values['1B'] == 1,
        values['2B'] == 1,
        values['3B'] == 1,
        values['SS'] == 1,
        values['OF'] == 3,
    ) + tuple([values['id%d' % i] <= 1 for i in range(len(items))])

    objective = lambda val: val[target] + delta * val['DK_Std']
    p = KSP(objective, items, goal='max', constraints=constraints)
    # requires cvxopt and glpk installed, see http://openopt.org/KSP for other
    r = p.solve('glpk', iprint=0)
    playerlist = r.xf
    print(playerlist)

    # r.xf is a list of players- we will merge their info back and return a DF
    df2 = df[df['player'].isin(playerlist)]
    # df2 is the latest lineup - we'll return the frame [for now]
    df2[['DK_Aggregate', 'DK_Rotogrind', 'DK_Sabersim', 'DK_Swish', 
	'DK_Numberfire','DK_Cafe','DK_Value','DK_Std','cruz','Cruz_Value']] = np.round(
        df2[['DK_Aggregate', 'DK_Rotogrind', 'DK_Sabersim', 'DK_Swish', 
		'DK_Numberfire','DK_Cafe','DK_Value','DK_Std','cruz','Cruz_Value']], 1)
    ajax = df2[['dk_pos',
                'player',
                'team',
                'matchup',
                'dk_sal',
		'cruz',
		'Cruz_Value',
                'DK_Aggregate',
                'DK_Std',
                'DK_Value',
		'DK_Rotogrind',
		'DK_Sabersim',
		'DK_Swish',
		'DK_Numberfire',
		'DK_Cafe']].to_json(orient='records')

    if(return_format == 'ajax'):
        return(ajax)
    else:
        return(df2)
