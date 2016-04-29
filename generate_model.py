# coding: utf-8
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import pickle
#from datetime import datetime, timedelta
import datetime
from bs4 import BeautifulSoup
import requests
import cnfg
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import defaultdict
from collections import OrderedDict
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from proj_elastic import InsertLogs
from sklearn import ensemble
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error


# In[2]:
config = cnfg.load("/home/ubuntu/dfsharp/.rotoguru_config")
url = config["url"]


def daily_download():
    # read in the user and key from config file
    # read in daily update of season long box scores
    df = pd.read_csv(url, sep=':')

    # format date as index, reset and sort old to new
    df.index = [pd.to_datetime(str(x), format='%Y%m%d') for x in df.Date]
    df.reset_index(inplace=True)
    df = df.sort(['index', 'Team'], ascending=[1, 1])

    # cut off note row
    df = df[1:]
    # rename some columns
    df = df.rename(columns={'H/A': 'home', 'First  Last': 'name', 'Team pts': 'team_pts',
                            'Opp pts': 'opp_pts', 'DK Sal': 'dk_sal', 'DK pos': 'dk_pos',
                            'DK Change': 'dk_change', 'GTime(ET)': 'gametime'})

    # only use these columns (for now)
    df = df[['index', 'GameID', 'gametime', 'name', 'Minutes', 'Start', 'active', 'DKP', 'Team', 'Opp',
             'home', 'team_pts', 'opp_pts', 'dk_sal', 'dk_pos', 'dk_change', 'Stats', 'DoubleD']]

    # only train on players who played > 0 minutes (keep today's players in
    # frame)
    today = datetime.datetime.today()
    df = df[(df['active'] > 0) | (df['index'] == today.strftime('%Y%m%d'))]

    return(df)


# In[3]:

def make_dvp(df):
    # create sportvu clusters
    cdf = pd.read_csv(
        '/home/ubuntu/dfsharp/sportvu_clusters.csv',
        sep='\t',
        encoding='utf-16')
    df = df.merge(cdf[['Name', 'Cluster Position Off', 'Cluster Position Def']].drop_duplicates(
        subset=['Name']), how='left', left_on='name', right_on='Name')
    df['Cluster Position Off'] = df[
        'Cluster Position Off'].fillna(df['dk_pos'])
    df['kpos'] = df['Cluster Position Off']
    return(df)


def add_pace(df):
    holl = pd.read_html('http://espn.go.com/nba/hollinger/teamstats')
    holl = holl[0][1:]
    holl.columns = holl.iloc[0]
    holl = holl.reindex(holl.index.drop(1))
    holl.set_value(3, 'TEAM', 'Oklahoma')
    # read in crosswalk
    teams = pd.read_csv('/home/ubuntu/dfsharp/team_crosswalk.csv', sep='\t')

    holl = pd.merge(
        left=holl,
        right=teams,
        left_on='TEAM',
        right_on='team_city')
    holl['team'] = holl['team'].str.lower()
    # merge team stats into real
    holl = holl[['team', 'team_city', 'PACE', 'AST', 'TO', 'ORR',
                 'DRR', 'REBR', 'EFF FG%', 'TS%', 'OFF EFF', 'DEF EFF']]
    holl = holl.rename(columns={'OFF EFF': 'OFF_EFF', 'DEF EFF': 'DEF_EFF'})
    holl['PACE'] = pd.to_numeric(holl['PACE'])
    holl['OFF_EFF'] = pd.to_numeric(holl['OFF_EFF'])
    holl['DEF_EFF'] = pd.to_numeric(holl['DEF_EFF'])

    hero_pace = holl[['team', 'PACE', 'OFF_EFF']]
    villain_pace = holl[['team', 'PACE', 'DEF_EFF']]
    villain_pace = villain_pace.rename(columns={'PACE': 'PACE_OPP'})

    df = pd.merge(
        left=df,
        right=hero_pace,
        left_on='Team',
        right_on='team',
        how='left')
    df = pd.merge(
        left=df,
        right=villain_pace,
        left_on='Opp',
        right_on='team',
        how='left')
    df['pace_sum'] = df['PACE'] + df['PACE_OPP']

    return(df)

# total active games per player


def active_games(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(350)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].active.sum()
# avg minutes past 7 days


def min_avg_7_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(7)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
# avg MPG when >0 past 90 days *used for dk/min


def min_avg_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
# dk pts scored past 90 days * used for dk/min


def dk_avg_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].DKP.mean()
# pts scored by team past 90 days [deprecated]
# def team_pts_90_days(x):
#    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index'] < x['index']) & (df['Team'] == x['Team'])]['team_pts'].mean()
# pts allowed by opponent past 90 days


def opp_pts_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) & (df['index']
                                                                 < x['index']) & (df['Opp'] == x['Opp'])]['team_pts'].mean()
# draftkings standard deviation!


def dk_std_90_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(90)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].DKP.std()
# draftkings local MAX


def dk_max_30_days(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(30)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].DKP.max()
# create avg minutes when starting


def min_when_starting(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(150)) & (df['Start'] == 1)
              & (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
# create avg minutes when starting


def min_when_bench(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(150)) & (df['Start'] == 0)
              & (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()


def starts_past_week(x):
    return df[(df['gp'] >= x['gp'] - 3) & (df['gp'] < x['gp'])
              & (df['name'] == x['name'])].Start.sum()
# if they're starting today, and they have <= 1 start in past 7 days, use
# min_when_start instead


def adjust_minutes(row):
    if (row['Start'] == True) and (row['starts_past_week'] <= 1) and (
            row['min_when_start'] > row['min_3g_avg']):
        return(row['min_when_start'])
    else:
        return(row['min_3g_avg'])
# create DKP allowed vs each position by team


def dvp(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(180)) & (df['index'] <
                                                                  x['index']) & (df['Opp'] == x['Opp']) & (df['kpos'] == x['kpos'])]['DKP'].mean()
# minutes yesterday


def min_yest(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(1)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].Minutes.mean()
# create back to back boolean


def create_b2b_bool(row):
    if row['min_yest'] > 30:
        return(1)
    else:
        return(0)
# 1) need Team MP, Team FGA, team FTA, team TOV for usage


def team_mp(x):
    return df[(df['index'] == x['index']) & (
        df['Team'] == x['Team'])]['Minutes'].sum()


def team_fga(x):
    return df[(df['index'] == x['index']) & (
        df['Team'] == x['Team'])]['fga'].sum()


def team_fta(x):
    return df[(df['index'] == x['index']) & (
        df['Team'] == x['Team'])]['fta'].sum()


def team_tov(x):
    return df[(df['index'] == x['index']) & (
        df['Team'] == x['Team'])]['tov'].sum()
# USAGE: 100 * ((FGA + 0.44 * FTA + TOV) * (Tm MP / 5)) / (MP * (Tm FGA +
# 0.44 * Tm FTA + Tm TOV)).


def usage(x):
    try:
        usage = 100 * ((x['fga'] + 0.44 * x['fta'] + x['tov']) * (x['team_mp'] / 5)) / (
            x['Minutes'] * (x['team_fga'] + 0.44 * x['team_fta'] + x['team_tov']))
    except ZeroDivisionError:
        usage = 0
    if (usage > 50):
        usage = 50
    return(usage)


def usage_3g_avg(x):
    return df[(df['gp'] >= x['gp'] - 3) & (df['gp'] < x['gp'])
              & (df['name'] == x['name'])].usage.mean()


def usage_5g_avg(x):
    return df[(df['gp'] >= x['gp'] - 5) & (df['gp'] < x['gp'])
              & (df['name'] == x['name'])].usage.mean()
# historical value


def value(x):
    val = x['DKP'] / (x['dk_sal'] / 1000)
    return(val)


def value_3g_avg(x):
    return df[(df['gp'] >= x['gp'] - 3) & (df['gp'] < x['gp'])
              & (df['name'] == x['name'])].value.mean()


def min_3g_avg(x):
    return df[(df['gp'] >= x['gp'] - 3) & (df['gp'] < x['gp'])
              & (df['name'] == x['name'])].Minutes.mean()


def starter_min(x):
    return df[(df['index'] == x['index']) & (
        df['Team'] == x['Team']) & df['Start'] == 1].Minutes.mean()


def starter_5g_avg(x):
    return df[(df['gp'] >= x['gp'] - 5) & (df['gp'] < x['gp'])
              & (df['name'] == x['name'])].starter_min.mean()
# minutes vs starters 5 game average


def mvs_5g_avg(x):
    return df[(df['gp'] >= x['gp'] - 5) & (df['gp'] < x['gp']) &
              (df['name'] == x['name'])].min_vs_starters.mean()
# add up their total double doubles


def dbl_dbl(x):
    return df[(df['index'] >= x['index'] - pd.DateOffset(350)) &
              (df['index'] < x['index']) & (df['name'] == x['name'])].DoubleD.sum()
# 3game avg of FGA


def fga_3g_avg(x):
    return df[(df['gp'] >= x['gp'] - 3) & (df['gp'] < x['gp'])
              & (df['name'] == x['name'])].fga.mean()
'''
add_stats- adds stats
    input: dataframe sorted ascending by dates
    outputs: same frame with added stat columns
'''


def add_stats(df):

    df['gp'] = df.apply(active_games, axis=1)
    df['min_3g_avg'] = df.apply(min_3g_avg, axis=1)

    #df['min_7d_avg'] = df.apply(min_avg_7_days, axis=1)
    df['min_90d_avg'] = df.apply(min_avg_90_days, axis=1)
    df['dk_avg_90_days'] = df.apply(dk_avg_90_days, axis=1)
    # df['teampts_avg'] = df.apply(team_pts_90_days, axis=1)
    # df['opppts_avg'] = df.apply(opp_pts_90_days, axis=1)
    df['dk_per_min'] = df['dk_avg_90_days'] / df['min_90d_avg']
    # transform DK points to more normal distro
    df['DKP_trans'] = df['DKP']**.5
    # create columns for - positive DK change; negative DK change
    # df['dk_sal_increase'] = np.where((df['dk_change'] > 0), True, False)
    # df['dk_sal_decrease'] = np.where((df['dk_change'] < 0), True, False)
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
    df['dvprank'] = pd.qcut(
        df['dvp'], [
            0.05, 0.1, 0.25, 0.5, 0.75, .93, 1], labels=False)
    # combine PACE and dvp
    df['pace_dvp'] = (df['pace_sum'] / 10) + df['dvp']

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
    df[['pts',
        'rbs',
        'stl',
        'ast',
        'blk',
        '3pm',
        'fgm',
        'fga',
        'ftm',
        'fta',
        'tov']] = df[['pts',
                      'rbs',
                      'stl',
                      'ast',
                      'blk',
                      '3pm',
                      'fgm',
                      'fga',
                      'ftm',
                      'fta',
                      'tov']].apply(lambda x: pd.to_numeric(x,
                                                            errors='coerce'))
    df[['pts', 'rbs', 'stl', 'ast', 'blk', '3pm', 'fgm',
        'fga', 'ftm', 'fta', 'tov']].fillna(0, inplace=True)

    # add yesterdays minutes
    df['min_yest'] = df.apply(min_yest, axis=1)
    # create back to back boolean column [over 30 minutes played the prior day]
    df['b2b'] = df.apply(create_b2b_bool, axis=1)

    # fillna just in case
    df['Minutes'] = df['Minutes'].fillna(value=0)
    df['fga'] = df['fga'].fillna(value=0)
    df['fta'] = df['fta'].fillna(value=0)
    df['tov'] = df['tov'].fillna(value=0)

    # add team stats for usage calc
    df['team_mp'] = df.apply(team_mp, axis=1)
    df['team_fga'] = df.apply(team_fga, axis=1)
    df['team_fta'] = df.apply(team_fta, axis=1)
    df['team_tov'] = df.apply(team_tov, axis=1)

    # add individual usage / 3 game rolling avg
    df['usage'] = df.apply(usage, axis=1)
    df['usage_3g_avg'] = df.apply(usage_3g_avg, axis=1)
    df['usage_5g_avg'] = df.apply(usage_5g_avg, axis=1)

    # add value / 3 game rolling avg for val
    df['value'] = df.apply(value, axis=1)
    df['value_3g_avg'] = df.apply(value_3g_avg, axis=1)

    # add starter min - average minutes played of all the starters
    df['starter_min'] = df.apply(starter_min, axis=1)

    # add game by game minutes vs starter average
    df['min_vs_starters'] = df['Minutes'] - df['starter_min']
    df['mvs_5g_avg'] = df.apply(mvs_5g_avg, axis=1)

    # add 3game average of starter minutes
    df['starter_5g_avg'] = df.apply(starter_5g_avg, axis=1)

    # add rolling avg of fga
    df['fga_3g_avg'] = df.apply(fga_3g_avg, axis=1)

    # add double double count
    df['dbl_dbl_cnt'] = df.apply(dbl_dbl, axis=1)
    # create "double double per game" stat
    df['dbl_dbl_per_game'] = df['dbl_dbl_cnt'] / df['gp']
    # combo stat: Minutes + FGA + dbl_dbl_per_game
    df['combo'] = df['min_proj'] + df['dbl_dbl_per_game'] + df['fga_3g_avg']

    return(df)

''' train_model - trains linear regression on given df
    inputs: df - dataframe to train on
            num - num to start slice at
    outputs: fitted model
    side-effects: prints summary statistics
                  pickles model
'''


def train_save_model(df, num=0, num2=20000):
    # train on most recent 30 days?
    #train = df[num:num2].dropna(subset=['DKP_trans','Start','dk_avg_90_days','home','dvp','usage_5g_avg','min_proj'])
    train = df[
        num:num2].dropna(
        subset=[
            'DKP',
            'Start',
            'dk_per_min',
            'pace_dvp',
            'combo'])
    Y_train, X_train = dmatrices('''DKP_trans ~  Start  + dk_per_min + pace_dvp + combo
                 ''', data=train, return_type='dataframe')

    model = sm.OLS(Y_train, X_train)
    results = model.fit()
    print(results.summary())
    path = '/home/ubuntu/dfsharp/latest_model.p'
    pickle.dump(results, open(path, "wb"))
    return(results)


def train_save_booster(df, num=0, num2=20000):
    # Load data
    df = df[
        num:num2].dropna(
        subset=[
            'DKP',
            'Start',
            'dk_per_min',
            'dvprank',
            'pace_sum',
            'min_proj',
            'home'])

    #X = scale(df[['Start','dk_per_min','pace_dvp','combo']])
    df['home'] = df['home'].map({'H': 1.0, 'A': 0.0})
    X = scale(df[['Start', 'dk_per_min', 'dvprank',
                  'pace_sum', 'min_proj', 'home']])
    X = X.astype(np.float32)
    y = df['DKP']

    # Fit regression model
    params = {'n_estimators': 450, 'max_depth': 2, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 0.3,
              'learning_rate': 0.1, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X, y)
    print(clf.score(X, y))
    path = '/home/ubuntu/dfsharp/latest_model.p'
    pickle.dump(clf, open(path, "wb"))
    return(clf)


# A) daily download
df = daily_download()

# B) create DVP
df = make_dvp(df)

# C) add pace
df = add_pace(df)

# D) add stats
df = add_stats(df)
# (optional) skip stat creation and load latest gamelogs df
#df = pd.read_csv('/home/ubuntu/dfsharp/gamelogs.csv')
print(len(df))
df.Date = df['GameID'].str[:8]

# E) pull out todays frame
today = datetime.datetime.today() - datetime.timedelta(hours=4)
#todays_players = df[df['index'] == today.strftime('%Y%m%d')]
todays_players = df[df.Date == today.strftime('%Y%m%d')]

print(len(todays_players))


csvpath = '/home/ubuntu/dfsharp/csvs/' + \
    today.strftime('%Y%m%d') + '_players.csv'
todays_players.to_csv(csvpath)
# F) insert gamelogs into elasticsearch, save to CSV
not_today = df[df['index'] != today.strftime('%Y%m%d')]
not_today.to_csv('gamelogs.csv')
InsertLogs(not_today, indexer="gamelogs")


# G) remove outliers for model training
df = df[df['dvp'] < 29]
df = df[df['dvp'] > 12]
df = df[df['dk_avg_90_days'] > 0]
df = df[df['Minutes'] > 5]
df = df[df['fga'] > 0]


# H) train and save the model
yo = train_save_booster(df, 5000, 20000)
