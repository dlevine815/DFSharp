# coding: utf-8
import pandas as pd
import json
from pprint import pprint
import urllib
import requests
from time import sleep
import re
from bs4 import BeautifulSoup
from selenium import webdriver
import numpy as np
from selenium.webdriver.support.ui import Select
import datetime

def get_fangraph_pitchers():
    # get al pitchers
    al = pd.read_html('http://www.fangraphs.com/dailyprojections.aspx?pos=all&stats=pit&type=sabersim&team=0&lg=al&players=0')
    fgpal = al[15]
    sleep(2)
    # get nl pitchers
    nl = pd.read_html('http://www.fangraphs.com/dailyprojections.aspx?pos=all&stats=pit&type=sabersim&team=0&lg=nl&players=0')
    fgpnl = nl[15]
    # merge and return
    fgp = fgpal.append(fgpnl)
    return(fgp)

def get_fangraph_batters():
    poslist = ['c','1b','2b','ss','3b','rf','cf','lf','dh']   
    df = pd.DataFrame()
    for pos in poslist:
        tmp = pd.read_html('http://www.fangraphs.com/dailyprojections.aspx?pos='+pos+'&stats=bat&type=sabersim&team=0&lg=al&players=0')
        df = df.append(tmp[15])
        sleep(2)
        tmp2 = pd.read_html('http://www.fangraphs.com/dailyprojections.aspx?pos='+pos+'&stats=bat&type=sabersim&team=0&lg=nl&players=0')
        df = df.append(tmp2[15])
        sleep(2)
    return(df)

def parse_json_stream(stream):
    decoder = json.JSONDecoder()
    while stream:
        obj, idx = decoder.raw_decode(stream)
        yield obj
        stream = stream[idx+1:].lstrip()

'''
gets projections from swish analytics
inputs- pitchers or batters
outputs- dataframe of projections

'''
def get_swish(pitchers=True):
    if pitchers==True:
        url = 'https://www.swishanalytics.com/optimus/mlb/dfs-pitcher-projections'
        data_cen = re.compile('this.pitcherArray = ')
    else:
        url = 'https://www.swishanalytics.com/optimus/mlb/dfs-batter-projections'
        data_cen = re.compile('this.batterArray = ')
        
    # soup object of site
    soup = BeautifulSoup(urllib.urlopen(url))
    data = soup.find("script", text=data_cen)
    array = re.match(r"[^[]*\[([^]]*)\]", str(data)).groups()[0]
    sw = pd.DataFrame()
    mygen = parse_json_stream(array)
    row = -1
    for i in mygen:
        row = row + 1
        for key in i.keys():
            sw.loc[row,key] = i[key]
            
    return(sw)


def make_fire_frame(html, site='FD_Numberfire'):
    soup = BeautifulSoup(html)
    
    # list of all batter names
    namelist = []
    for tag in soup.find_all('td', {'class': 'player'}):
        namefull = tag.text.lstrip().rstrip()
        namefinal = namefull.split('(')
        namelist.append(namefinal[0].rstrip())
       
    # list of their dk proj
    dklist = []
    for tag in soup.find_all('td', {'class': 'sep nf col-fp'}):
        dklist.append(tag.text.lstrip().rstrip())
        
    
    nf = pd.DataFrame({'name' : namelist,
                       site : dklist})
    
    return(nf)


# In[7]:

def get_numberfire(pitchers=True):
    if pitchers==True:
        url = 'http://www.numberfire.com/mlb/daily-fantasy/daily-baseball-projections/pitchers'
    else:
        url = 'http://www.numberfire.com/mlb/daily-fantasy/daily-baseball-projections'
            
    driver = webdriver.PhantomJS()
    driver.set_window_size(1120, 550)
    driver.get(url)
    
    # make fanduel frame
    fdf = make_fire_frame(driver.page_source)
    # change dfs site list to DK [fanduel is default]
    Select(driver.find_element_by_css_selector("select#dfs-site-list")).select_by_value('4')
    # wait for page to load
    sleep(5)
    # grab html
    html = driver.page_source
    # make draftkings frame
    dkf = make_fire_frame(html, site='DK_Numberfire')
    # quit driver
    driver.quit()
    # merge the two frames and return
    nf = pd.merge(fdf, dkf, on='name', how='inner')
    return(nf)
    


# In[8]:

# takes a json formatted as string, returns dataframe
def build_cafe_frame(jsonobj):
    
    names = []
    dk_proj = []
    fd_proj = []
    
    mygen = parse_json_stream(jsonobj)
    for i in mygen:
        # append name
        names.append(i['name'])
        # append projections
        dk_proj.append(i['projections']['draftkings'])
        fd_proj.append(i['projections']['fanduel'])
    
    cf = pd.DataFrame({'name' : names,
                   'DK_Cafe' : dk_proj,
                   'FD_Cafe' : fd_proj
                  })
    return(cf)


# In[9]:

# no inputs
# returns tuple of dataframes, first is batters 2nd is pitchers
def get_fantasycafe():
    # soup object of site
    url = 'https://www.dailyfantasycafe.com/tools/projections/mlb'
    soup = BeautifulSoup(urllib.urlopen(url))
    
    # all data is stored in 'projections-tool'
    data = soup.find('div', {'id': 'projections-tool'})
    # fix idiot quotes
    data2 = re.sub(r'&ldquo;|&rdquo;|&quot;', '"', str(data))
    # grabbing batter data
    batters = re.match(r"[^[]*\[([^]]*)\]", data2).groups()[0]
    # split on open bracket
    datasplits = data2.split('[')
    # grabs the json object of pitcher data
    pitchers = datasplits[3].split(']')[0]
    # batter frame
    fcb = build_cafe_frame(batters)
    # pitcher frame
    fcp = build_cafe_frame(pitchers)
    
    return(fcb, fcp)  


# In[10]:

def get_rotogrinders():
    # pitchers
    rgdkp = pd.read_csv('https://rotogrinders.com/projected-stats/mlb-pitcher.csv?site=draftkings')
    rgfdp = pd.read_csv('https://rotogrinders.com/projected-stats/mlb-pitcher.csv?site=fanduel')
    # hitters
    rgdkh = pd.read_csv('https://rotogrinders.com/projected-stats/mlb-hitter.csv?site=draftkings')
    rgfdh = pd.read_csv('https://rotogrinders.com/projected-stats/mlb-hitter.csv?site=fanduel')
    # rename columns
    rgdkp = rgdkp.rename(columns={'fpts': 'DK_Rotogrind', 'salary': 'dk_sal', 'pos': 'dk_pos'})
    rgfdp = rgfdp.rename(columns={'fpts': 'FD_Rotogrind', 'salary': 'fd_sal', 'pos': 'fd_pos'})
    rgdkh = rgdkh.rename(columns={'fpts': 'DK_Rotogrind', 'salary': 'dk_sal', 'pos': 'dk_pos'})
    rgfdh = rgfdh.rename(columns={'fpts': 'FD_Rotogrind', 'salary': 'fd_sal', 'pos': 'fd_pos'})
    # merge
    rgp = pd.merge(rgdkp, rgfdp, on='player', how='inner')
    rgh = pd.merge(rgdkh, rgfdh, on='player', how='inner')
    
    rgp = rgp[['player','team_x','dk_sal','dk_pos','DK_Rotogrind','fd_sal','fd_pos','FD_Rotogrind']]
    rgh = rgh[['player','team_x','dk_sal','dk_pos','DK_Rotogrind','fd_sal','fd_pos','FD_Rotogrind']]
    
    return(rgp, rgh)


# In[11]:

# no inputs
# returns tuple of dataframes, first is batters 2nd is pitchers
def get_roto_full():
    # soup object of site
    urlb = 'https://rotogrinders.com/projected-stats/mlb-hitter?site=fanduel'
    batters = get_roto_json(urlb)
    # batter frame
    rgb2 = build_roto_batters(batters)
    
    # soup object of site
    urlp = 'https://rotogrinders.com/projected-stats/mlb-pitcher?site=fanduel'
    pitchers = get_roto_json(urlp)
    # batter frame
    rgp2 = build_roto_pitchers(pitchers)    
    
    return(rgp2, rgb2)


# In[12]:

def get_roto_json(url):
    soup = BeautifulSoup(urllib.urlopen(url))
    # all data is stored in a json in a script
    data = soup.find_all('script')
    playerjson = str(data).split(' data = ')
    playerjson2 = playerjson[1].split('];\n')
    players = playerjson2[0][1:]
    return(players)


# In[13]:

# takes a json formatted as string, returns dataframe
def build_roto_batters(jsonobj):
    
    names = []
    ou = []
    line = []
    total = []
    delta = []
    order = []
    confirmed = []
    woba = []
    ops  = []
    iso = []
    vspitchname = []
    vspitchhand = []
    ab = []

    mygen = parse_json_stream(jsonobj)
    for i in mygen:
        name = i['player']['first_name'] + ' ' + i['player']['last_name']
        names.append(name)
        ou.append(i['o/u'])
        line.append(i['line'])
        total.append(i['total'])
        #delta.append(i['delta'])
        if 'order' in i.keys():
            order.append(i['order'])
        else:
            order.append(0)
        if 'confirmed' in i.keys():
            confirmed.append(i['confirmed'])
        else:
            confirmed.append(False)
        ab.append(i['ab'])
        if int(i['ab']) > 0:
            woba.append(i['woba'])
            ops.append(i['ops'])
            iso.append(i['iso'])
        else:
            woba.append(0)
            ops.append(0)
            iso.append(0)
        pname = i['pitcher']['first_name'] + ' ' + i['pitcher']['last_name']
        vspitchname.append(pname)
        vspitchhand.append(i['pitcher']['hand'])

    rf = pd.DataFrame({'name' : names,
                           'ou' : ou,
                           'line' : line,
                            'total' : total,
                           # 'delta' : delta,
                           'order' : order,
                           'confirmed' : confirmed,
                           'woba' : woba,
                           'ops' : ops,
                           'iso' : iso,
                           'vsp' : vspitchname,
                           'vshand' : vspitchhand,
                           'ab' : ab
                      })
    return(rf)


# In[14]:

# takes a json formatted as string, returns dataframe
def build_roto_pitchers(jsonobj):
    
    names = []
    ou = []
    line = []
    total = []
    delta = []
    hand = []
    '''lwoba = []
    rwoba = []
    lslga = []
    rslga = []
    liso = []
    riso = []
    lk9 = []
    rk9 = []'''
    gp = []

    mygen = parse_json_stream(jsonobj)
    for i in mygen:
        name = i['player']['first_name'] + ' ' + i['player']['last_name']
        names.append(name)
        ou.append(i['o/u'])
        line.append(i['line'])
        total.append(i['total'])
        #delta.append(i['delta'])
        hand.append(i['player']['hand'])
        gp.append(i['gp'])
        '''
        if int(i['gp']) > 0:
            lwoba.append(i['lwoba']) 
            rwoba.append(i['rwoba'])
            lslga.append(i['lslga'])
            rslga.append(i['rslga'])
            liso.append(i['liso'])
            riso.append(i['riso'])
            lk9.append(i['lk/9'])
            rk9.append(i['rk/9'])
        else:
            lwoba.append(0)
            rwoba.append(0)
            lslga.append(0)
            rslga.append(0)
            liso.append(0)
            riso.append(0)
            lk9.append(0)
            rk9.append(0)
        '''


    rf = pd.DataFrame({'name' : names,
                       'ou' : ou,
                       'line' : line,
                        'total' : total,
                        #'delta' : delta,
                       'hand' : hand,
                       ''''lwoba': lwoba,
                       'rwoba': rwoba,
                       'lslga': lslga,
                       'rslga': rslga,
                       'liso': liso,
                       'riso': riso,
                       'lk9': lk9,
                       'rk9': rk9,'''
                       'gp': gp
                      })
    return(rf)


# In[15]:

def merge_players(rgh, fgh, swh, nfh, fch, rgh2, site='dk', pitchers=False):
    # rename columns
    fgh['DK_Sabersim'] = fgh['DraftKings']
    fgh['FD_Sabersim'] = fgh['FanDuel']
    swh['DK_Swish'] = swh['dk_pts']
    swh['FD_Swish'] = swh['fd_pts']
    
    # merge rotogrinders and fangraphs/sabersim
    df2 = pd.merge(rgh, fgh, left_on='player', right_on='Name', how='inner')
    
    # merge in swish
    if pitchers==False:
        swh = swh[['player_name','matchup','team_short','time','DK_Swish','FD_Swish',
              'dk_salary','fd_salary','fd_avg','dk_avg','bats']]
    else:
        swh = swh[['player_name','matchup','team_short','DK_Swish','FD_Swish',
              'dk_salary','fd_salary','fd_avg','dk_avg']]
        
    df3 = pd.merge(df2, swh, left_on='player', right_on='player_name', how='inner')
    
    # merge in numberfire
    df4 = pd.merge(df3, nfh, left_on='player', right_on='name', how='inner')
    # merge in fantasycafe
    df5 = pd.merge(df4, fch, left_on='player', right_on='name', how='inner')
    
    # merge in RG2
    df5 = pd.merge(df5, rgh2, left_on='player', right_on='name', how='inner')
    
    # convert columns to numeric
    df5 = df5.convert_objects(convert_numeric=True) 
    # rename team
    df5 = df5.rename(columns={'team_x': 'team'})
    
    if site=='dk':
        # create aggregate
        df5['DK_Aggregate'] = ( df5['DK_Rotogrind'] + df5['DK_Sabersim'] + 
                                df5['DK_Swish'] + df5['DK_Numberfire'] + df5['DK_Cafe'] ) / 5
        df5['DK_plusminus'] = df5['DK_Aggregate'] - df5['dk_avg']
        # create value, standard dev, ceiling and floor
        df5['DK_Value'] = df5['DK_Aggregate'] / (df5['dk_sal'] / 1000)
        df5['DK_Std'] = df5[['DK_Rotogrind','DK_Sabersim','DK_Swish','DK_Numberfire','DK_Cafe']].std(axis=1).round(3)
        df5['DK_Ceiling'] = df5['DK_Aggregate'] + (2*df5['DK_Std'])
        df5['DK_Floor'] = df5['DK_Aggregate'] - (2*df5['DK_Std'])
        if pitchers==False:
            dk = df5[['player','time','team','matchup','bats','vshand','vsp','order'
                      ,'line','ou','total'
                     ,'dk_sal','dk_pos','dk_avg','DK_Rotogrind','DK_Sabersim'
                     ,'DK_Swish','DK_Numberfire','DK_Cafe','DK_Aggregate'
                     ,'DK_Value','DK_Floor','DK_Ceiling','DK_Std']]
        else:
            dk = df5[['player','team','matchup','hand','line','ou','total'
                     ,'dk_sal','dk_pos','dk_avg','DK_Rotogrind','DK_Sabersim'
                     ,'DK_Swish','DK_Numberfire','DK_Cafe','DK_Aggregate'
                     ,'DK_Value','DK_Floor','DK_Ceiling','DK_Std'
                    # ,'lwoba','rwoba','lk9','rk9'
                     ]]      
        tmp = dk.select_dtypes(include=[np.number])
        dk.loc[:, tmp.columns] = np.round(tmp, 2)
        return(dk)
    else:
        # fanduel aggregate
        df5['FD_Aggregate'] = ( df5['FD_Rotogrind'] + df5['FD_Sabersim'] + 
                                df5['FD_Swish'] + df5['FD_Numberfire'] + df5['FD_Cafe'] ) / 5
        df5['FD_plusminus'] = df5['FD_Aggregate'] - df5['fd_avg']
        # fd value
        df5['FD_Value'] = df5['FD_Aggregate'] / (df5['fd_sal'] / 1000)    
        df5['FD_Std'] = df5[['FD_Rotogrind','FD_Sabersim','FD_Swish','FD_Numberfire','FD_Cafe']].std(axis=1).round(3)
        df5['FD_Ceiling'] = df5['FD_Aggregate'] + (2*df5['FD_Std'])
        df5['FD_Floor'] = df5['FD_Aggregate'] - (2*df5['FD_Std'])
        if pitchers==False:
            fd = df5[['player','time','team','matchup','bats','vshand','vsp','order'
                      ,'line','ou','total'
                     ,'fd_sal','fd_pos','fd_avg','FD_Rotogrind','FD_Sabersim'
                     ,'FD_Swish','FD_Numberfire','FD_Cafe','FD_Aggregate'
                     ,'FD_Value','FD_Floor','FD_Ceiling','FD_Std']]
        else:
            fd = df5[['player','team','matchup','hand','line','ou','total'
                     ,'fd_sal','fd_pos','fd_avg','FD_Rotogrind','FD_Sabersim'
                     ,'FD_Swish','FD_Numberfire','FD_Cafe','FD_Aggregate'
                     ,'FD_Value','FD_Floor','FD_Ceiling','FD_Std'
                    # ,'lwoba','rwoba','lk9','rk9'
                     ]]
        tmp = fd.select_dtypes(include=[np.number])
        fd.loc[:, tmp.columns] = np.round(tmp, 2)
        return(fd)  


# In[16]:

def get_aggregates(site='dk'):
    rg = get_rotogrinders()
    # rotogrinders pitchers
    rgp = rg[0]
    # rotogrinders hitters
    rgh = rg[1]
    
    # get fangraph pitchers
    fgp = get_fangraph_pitchers()
    # fangraph hitters
    fgh = get_fangraph_batters()
    
    # swish pitchers
    swp = get_swish()
    # swish hitters
    swh = get_swish(pitchers=False)
    
    # numberfire hitters
    nfh = get_numberfire(pitchers=False)
    # numberfire pitchers
    nfp = get_numberfire(pitchers=True)
    
    fc = get_fantasycafe()
    # fantasycafe hitters
    fch = fc[0]
    # fantasycafe pitchers
    fcp = fc[1]
    
    rg2 = get_roto_full()
    # rotogrinders full pitchers
    rgp2 = rg2[0]
    # rotogrinders full batters
    rgh2 = rg2[1]
    
    hitters = merge_players(rgh, fgh, swh, nfh, fch, rgh2, site=site, pitchers=False)
    pitchers = merge_players(rgp, fgp, swp, nfp, fcp, rgp2, site=site, pitchers=True)
    
    return(hitters, pitchers)



# get projections and write to csv
agg = get_aggregates(site='dk')
hitters = agg[0]
pitchers = agg[1]

today = datetime.datetime.today() - datetime.timedelta(hours=4)
path = '/home/ubuntu/dfsharp/mlb_dfs/projections/'+today.strftime('%Y%m%d')

hitters.to_csv(path+'_hitters.csv')
pitchers.to_csv(path+'_pitchers.csv')

# write to c

