# coding: utf-8

# In[70]:

import pandas as pd
import matplotlib as plt
# import seaborn as sns
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
import cnfg
pd.set_option("display.max_rows",300)

#get_ipython().magic(u'matplotlib inline')


# In[2]:


config = cnfg.load("/home/ubuntu/dfsharp/.rotoguru_config")
url = config["url"]


# In[3]:

def daily_download():
    # read in the user and key from config file   
    # read in daily update of season long box scores
    df = pd.read_csv(url, sep=':')
    
    # format date as index, reset and sort old to new
    df.index = [pd.to_datetime(str(x), format='%Y%m%d') for x in df.Date]
    df.reset_index(inplace=True)
    df = df.sort(['index', 'Team'], ascending=[1,1])
    
    # cut off note row
    df = df[1:]
    # rename some columns
    df = df.rename(columns={'H/A': 'home', 'First  Last': 'name', 'Team pts':'team_pts', 
                        'Opp pts': 'opp_pts','DK Sal':'dk_sal','DK pos':'dk_pos',
                       'DK Change':'dk_change','GTime(ET)':'gametime'})
    
    # only use these columns (for now)
    df = df[['index','GameID','gametime','name','Minutes','Start','active','DKP','Team','Opp',
         'home','team_pts','opp_pts','dk_sal','dk_pos','dk_change','Stats','DoubleD']]
    
    # only train on players who played > 0 minutes (keep today's players in frame)
    today = datetime.today()
    df = df[(df['Minutes'] > 0) | (df['index'] == today.strftime('%Y%m%d'))]



    
    return(df)

def make_dvp(df):
    # create sportvu clusters
    cdf = pd.read_csv('/home/ubuntu/dfsharp/sportvu_clusters.csv', sep='\t', encoding='utf-16')
    df = df.merge(cdf[['Name','Cluster Position Off','Cluster Position Def']].drop_duplicates(subset=['Name']), how='left', left_on='name', right_on='Name')
    df['Cluster Position Off'] = df['Cluster Position Off'].fillna(df['dk_pos'])
    df['kpos'] = df['Cluster Position Off']
    return(df)
# In[4]:

# make these into 1 function that takes- DF, DateNum, ColToAvg
# avg minutes past 7 days
def min_avg_7_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(7)) & (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
# avg MPG when >0 past 90 days *used for dk/min
def min_avg_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
# dk pts scored past 90 days * used for dk/min
def dk_avg_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index'] < x['index']) & (df['name'] == x['name'])].DKP.mean()
# pts scored by team past 90 days [deprecated]
#def team_pts_90_days(x):
#    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index'] < x['index']) & (df['Team'] == x['Team'])]['team_pts'].mean()
# pts allowed by opponent past 90 days
def opp_pts_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index'] < x['index']) & (df['Opp'] == x['Opp'])]['team_pts'].mean()
# draftkings standard deviation!
def dk_std_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index'] < x['index']) & (df['name'] == x['name'])].DKP.std()
# draftkings local MAX
def dk_max_30_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(30)) & (df['index'] < x['index']) & (df['name'] == x['name'])].DKP.max()

# create avg minutes when starting
def min_when_starting(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(150)) & (df['Start'] == 1) & (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
# create avg minutes when starting
def min_when_bench(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(150)) & (df['Start'] == 0) & (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
def starts_past_week(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(7)) & (df['index'] < x['index']) & (df['name'] == x['name'])].Start.sum()

# if they're starting today, and they have <= 1 start in past 7 days, use min_when_start instead
def adjust_minutes(row):
    if (row['Start'] == True) and (row['starts_past_week'] <= 1) and (row['min_when_start'] > row['min_7d_avg']):
        return(row['min_when_start'])
    else:
        return(row['min_7d_avg'])
# create DKP allowed vs each position by team
def dvp(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(180)) & (df['index'] < x['index']) & (df['Opp'] == x['Opp']) & (df['kpos'] == x['kpos'])]['DKP'].mean()
   

# In[5]:

# player's total double doubles [NOT WORKING]
#def opp_pts_90_days(x):
#    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index'] < x['index']) & (df['Opp'] == x['Opp'])]['team_pts'].mean()


# In[6]:

''' add_stats- adds stats
    input: dataframe sorted ascending by dates
    outputs: same frame with added stat columns
'''
def add_stats(df):

    df['min_7d_avg'] = df.apply(min_avg_7_days, axis=1)
    df['min_90d_avg'] = df.apply(min_avg_90_days, axis=1)
    df['dk_avg_90_days'] = df.apply(dk_avg_90_days, axis=1)
    # df['teampts_avg'] = df.apply(team_pts_90_days, axis=1)
    df['opppts_avg'] = df.apply(opp_pts_90_days, axis=1)
    df['dk_per_min'] = df['dk_avg_90_days'] / df['min_90d_avg']
    # transform DK points to more normal distro
    df['DKP_trans'] = df['DKP']**.5
    # create columns for - positive DK change; negative DK change
    df['dk_sal_increase'] = np.where((df['dk_change'] > 0), True, False)
    df['dk_sal_decrease'] = np.where((df['dk_change'] < 0), True, False)
    # create standard dev and max columns
    df['dk_std_90_days'] = df.apply(dk_std_90_days, axis=1)
    df['dk_max_30_days'] = df.apply(dk_max_30_days, axis=1)

    # get min when starting / bench
    df['min_when_start'] = df.apply(min_when_starting, axis=1)
    df['min_when_bench'] = df.apply(min_when_bench, axis=1)
    # count games started in past week
    df['starts_past_week'] = df.apply(starts_past_week, axis=1)
    # adjust minutes
    df['min_proj'] = df.apply(adjust_minutes, axis=1)
    # add dvp
    df['dvp'] = df.apply(dvp, axis=1)
    # add dvp rank
    df['dvprank'] = pd.qcut(df['dvp'], [0.05, 0.1, 0.25, 0.5, 0.75, .93, 1], labels=False)
    
    # create summary stats
    df['pts'] = df['Stats'].str.extract('(\d*)pt')
    df['rbs'] = df['Stats'].str.extract('(\d*)rb')
    df['stl'] = df['Stats'].str.extract('(\d*)st')
    df['ast'] = df['Stats'].str.extract('(\d*)as')
    df['blk'] = df['Stats'].str.extract('(\d*)bl')
    df['3pm'] = df['Stats'].str.extract('(\d*)trey')
    df['fgm'] = df['Stats'].str.extract('(\d*)-\d*fg')
    df['fga'] = df['Stats'].str.extract('\d*-(\d*)fg')
    df['ftm'] = df['Stats'].str.extract('(\d*)-\d*ft')
    df['fta'] = df['Stats'].str.extract('\d*-(\d*)ft')
    df['tov'] = df['Stats'].str.extract('(\d*)to')
    df[['pts','rbs','stl','ast','blk','3pm','fgm','fga','ftm','fta','tov']] = df[['pts','rbs','stl','ast','blk','3pm','fgm','fga','ftm','fta','tov']].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df[['pts','rbs','stl','ast','blk','3pm','fgm','fga','ftm','fta','tov']].fillna(0, inplace=True)

    #df.to_csv('/home/ubuntu/dfsharp/gamelogs/20160326_gamelogs.csv')
    
    return(df)


# In[65]:

''' train_model - trains linear regression on given df
    inputs: df - dataframe to train on
            num - num to start slice at
    outputs: fitted model
    side-effects: prints summary statistics
                  pickles model
'''
def train_save_model(df, num=0):
    # train on most recent 30 days?
    train = df[num:].dropna(subset=['DKP_trans','Start','min_proj','dk_per_min','home','dvp'])  
    Y_train, X_train = dmatrices('''DKP_trans ~  Start + min_proj  + dk_per_min + dvp
                                 + home   
                                 
                 ''', data=train, return_type='dataframe')
    
    model = sm.OLS(Y_train, X_train)
    results = model.fit()
    print(results.summary())
    path = '/home/ubuntu/dfsharp/latest_model.p'
    pickle.dump(results, open(path, "wb") )
    return(results)


# In[11]:


# A) daily download 
df = daily_download()

# B) create DVP
df = make_dvp()

# C) add stats
df = add_stats(df)

# D) pull out todays frame
today = datetime.today()
todays_players = df[df['index'] == today.strftime('%Y%m%d')]
csvpath = '/home/ubuntu/dfsharp/csvs/'+today.strftime('%Y%m%d')+'_players.csv'
todays_players.to_csv(csvpath)

# E) train and save model
train_save_model(df, 14000)