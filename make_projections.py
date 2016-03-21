
# coding: utf-8

# In[22]:

import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import pickle
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import defaultdict
from collections import OrderedDict
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
pd.set_option("display.max_rows",300)

get_ipython().magic(u'matplotlib inline')


# In[ ]:


'''
it loads latest model and todays_players CSV, dynamically determines
starters, generates today's projections, and pushes them up to elasticsearch

---
also generates today's optimal lineup and pushes THAT to elasticsearch
'''


# In[14]:

def soup_url(url):
    site = requests.get(url)
    page = site.text
    soup = BeautifulSoup(page)
    return soup

def make_depth_dict():
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    url = 'http://basketball.realgm.com/nba/depth-charts'
    soup = soup_url(url)
    keys = []
    options = soup.find_all(class_='ddl')
    teams = options[1].find_all('option')[1:]
    i = 0
    for team in teams:
        #temp = process.extractOne(team.text, team_dict.keys())
        temp = process.extractOne(team.text, team_names)
        #key = team_dict[temp[0]]
        #keys.append(key)
        keys.append(temp[0])
    data = soup.find_all('table', class_="basketball")
    depth_dict = {}
    for datum in data:
        starters = datum.find_all(class_='depth_starters')
        starter_links = starters[0].find_all('a')
        starting5 = []
        roster = defaultdict(str)
        for starter in starter_links:
            starting5.append([starter.text])
        depth = dict(zip(positions, starting5))
        starting5 = np.ravel(starting5)
        starting_lineup = dict(zip(starting5, positions))
        subs = datum.find_all(class_='depth_rotation')
        rotation = defaultdict(str)
        for sub in subs:
            for pos in positions:
                links = sub.find_all('td', {'data-th': pos})
                for link in links:
                    if link.find('a'):
                        depth[pos].append(link.find('a').text)
                        rotation[link.find('a').text] = pos
                        
        scrubs = datum.find_all(class_="depth_limpt")
        scrub_dict = defaultdict(str)
        for scrub in scrubs:
            for pos in positions:
                    links = scrub.find_all('td', {'data-th': pos})
                    for link in links:
                        if link.find('a'):
                            depth[pos].append(link.find('a').text)
                            scrub_dict[link.find('a').text] = pos
        for player, position in starting_lineup.iteritems():
            roster[player] = position
        for player, position in rotation.iteritems():
            roster[player] = position
        for player, position in scrub_dict.iteritems():
            roster[player] = position
        depth_dict[keys[i]] = {'roster': roster,  'depth': depth, 'starters': starting_lineup, 'rotation': rotation, 'scrubs': scrub_dict}
        i += 1
    #depth_dict['PHO'] = depth_dict['PHX']
    return depth_dict


# In[8]:

""" init starters:
        grabs current rosters from realGM
    inputs: dataframe of today's players
    outputs: df with 1 on all players who fuzzy match a starter in realGM
"""
def init_starters(df):
    depth = make_depth_dict()
    starters = []
    # append starters for each pos to empty starters list [in linear time?!]
    for i in depth:
        starters.append(depth[i]['starters'].keys()[0])
        starters.append(depth[i]['starters'].keys()[1])
        starters.append(depth[i]['starters'].keys()[2])
        starters.append(depth[i]['starters'].keys()[3])
        starters.append(depth[i]['starters'].keys()[4])    
        
    #starters  = list of starters obtained from RealGM
    # name - name in row
    def starter_match(name):
        top = process.extractOne(name, starters)
        if top[1] > 85:
            # print(top[0])
            return(True)
        else:
            return(False)
         
    df['Start_Raw'] = df['name'].isin(starters)
    df['Start'] = df['name'].apply(starter_match)
    return(df)


# In[9]:

''' project today - makes projections for today's games
    inputs:
        - trained model
        - df containing today's players
        
    outputs:
        - df with added projections and value cols
'''
def project_today(model, df):
    today_df = df[['dk_pos','dk_sal','Team','name','Start','min_7d_avg','dk_avg_90_days','dk_std_90_days','dk_max_30_days','home','dk_per_min','opppts_avg']].dropna()
    
    # add intercept and convert all to numeric
    Y_fake, features_real = dmatrices('''dk_sal ~ Start  + min_7d_avg + dk_per_min 
                                 + home + opppts_avg 
                 ''', data=today_df, return_type='dataframe')
    
    # MAKE LIVE PROJECTIONS <3
    today_df['DK_Proj'] = (model.predict(features_real, transform=False))
    today_df['DK_Proj'] = today_df['DK_Proj']**2
    
    today_df['proj_pure'] = today_df['min_7d_avg'] * today_df['dk_per_min']
    
    today_df['value'] = today_df['DK_Proj'] / (today_df['dk_sal'] / 1000) 
    today_df['ceiling'] = today_df['DK_Proj'] + today_df['dk_std_90_days']
    return(today_df)


# In[18]:

# (load team name list)
team_walk = pd.read_csv('/Users/shermanash/ds/nba/dfs_twitter/team_crosswalk.csv', sep='\t')
team_names = team_walk.team_long.tolist()


# In[6]:

# load latest model
path = '/Users/shermanash/ds/dfsharp_test/latest_model.p'
model = pickle.load( open( path, "rb" ) )


# In[12]:

# get DF of todays players
today = datetime.today()
filename = today.strftime('%Y%m%d')+'_players.csv'
todays_players = pd.read_csv(filename)


# In[23]:

# 6) generate today's starters
today_df = init_starters(todays_players)
# 7) generate today's projections
today_proj = project_today(model, today_df)
# 8) push timestamped projections to elasticsearch

# *format for csv
# hio = today_proj[['dk_pos','name','dk_sal','Start','DK_Proj']].to_csv('20160320_opt.csv', index=False)


# In[24]:

today_proj


# In[ ]:

'''

optimize function should eventually take as input
    -number of lineups requested
    -adjustments dict??
    -STAT to OPTIMIZE on!
'''

