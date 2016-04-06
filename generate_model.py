
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
    train = df[num:].dropna()  
    Y_train, X_train = dmatrices('''DKP_trans ~ Start  + min_7d_avg + dk_per_min 
                                 + home  + opppts_avg 
                                 
                 ''', data=train, return_type='dataframe')
    
    model = sm.OLS(Y_train, X_train)
    results = model.fit()
    print(results.summary())
    path = '/home/ubuntu/dfsharp/latest_model.p'
    pickle.dump(results, open(path, "wb") )
    return(results)


# In[8]:

# input - trained model - DF
# output - DF of yesterdays projections
def assess_yesterday(model, df):
    yest = datetime.today() - timedelta(days=1)
    yest_players = df[df['index'] == yest.strftime('%Y%m%d')]
    yest_players.dropna(inplace=True)  
    
    Y_yest, X_yest = dmatrices('''DKP_trans ~ Start  + min_7d_avg + dk_per_min 
                                 + home + opppts_avg 
                 ''', data=yest_players, return_type='dataframe')
    
    pred = yest_players[['index','name','dk_sal','Start','Minutes','min_7d_avg','dk_per_min','Opp','home','opppts_avg','DKP']]
    pred['DKP_Proj'] = (model.predict(X_yest, transform=False))
    pred['DKP_Proj'] = pred['DKP_Proj']**2
    
    pred['diff'] = pred['DKP'] - pred['DKP_Proj']
    pred['value'] = pred['DKP_Proj'] / (pred['dk_sal'] / 1000)
    pred['Mindiff'] = pred['Minutes'] - pred['min_7d_avg']
    return(pred)


# In[11]:


# 1) daily download 
df = daily_download()

# 2) add stats
df = add_stats(df)
# 3) pull out todays frame
today = datetime.today()
todays_players = df[df['index'] == today.strftime('%Y%m%d')]
csvpath = '/home/ubuntu/dfsharp/csvs/'+today.strftime('%Y%m%d')+'_players.csv'
todays_players.to_csv(csvpath)

# 4) train and save model
train_save_model(df, 10000)

# 5) assess yesterday's predictions (OPT)
# yest = assess_yesterday(model, df)

# 1-5 happen daily at 1 PM: the today's players CSV is saved to CSV, and the model is pickled to EC2
# yesterdays projections are pushed to elasticsearch once daily on their own index
