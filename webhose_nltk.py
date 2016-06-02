import unirest
import time
import webhose
from elasticsearch import Elasticsearch
import json
import cnfg
import pandas as pd
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
import datetime


config = cnfg.load("/home/ubuntu/dfsharp/.webhoser_config")
tok = config["token"]
webhose.config(token=tok)

# In[3]:


def get_archive():
    yesterday = datetime.date.today() - datetime.timedelta(hours=24)
    unix_time = yesterday.strftime("%s")
    # get response from webhose query 
    # response = unirest.get("https://webhose.io/search?token=" + tok + "&format=json&q=NBA+DFS",
    response = unirest.get("https://webhose.io/search?token=" + tok + "&format=json&q=MLB%20DFS",
    # response = unirest.get("https://webhose.io/search?token=" + tok + "&format=json&q=MLB+DFS+ts="+unix_time,

                           headers={
                               "Accept": "application/json"
                           }
                           )

    return(response)


# In[4]:

def get_playerlist():
    ''' old method for nba-
    df = pd.read_csv('/home/ubuntu/dfsharp/playernames.csv')
    playerlist = df['allplayers.text']
    '''
    # new method for mlb
    today = datetime.datetime.today() - datetime.timedelta(hours=4)
    yesterday = today - datetime.timedelta(hours=28)

    try:
	hit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + today.strftime('%Y%m%d') + '_hitters.csv')
	pit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + today.strftime('%Y%m%d') + '_pitchers.csv')
    except IOError:
	hit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + yesterday.strftime('%Y%m%d') + '_hitters.csv')
	pit = pd.read_csv('/home/ubuntu/dfsharp/mlb_dfs/projections/' + yesterday.strftime('%Y%m%d') + '_pitchers.csv')

    df = hit['player']
    playerlist = df.append(pit['player'])

    plist = []
    for p in playerlist:
        plist.append(p)

    list6 = [" ".join(n.split(", ")[::-1]) for n in plist]
    return(list6)


# In[5]:

''' inputs - sentlist - tokenized list of sentences
            article - original article
    outputs - dataframe mapping players to sentence bigrams
'''


def parse_article(sentlist, article):
    # get list of players
    playerlist = get_playerlist()
    # init dataframe
    newsframe = pd.DataFrame(columns=['sentid', 'player', 'text', 'text2'])

    sentid = 0
    for n, i in enumerate(sentlist):
        for p in playerlist:
            if p in i:
                try:
                    newsframe.loc[sentid] = pd.Series(
                        {'player': p, 'text': i, 'text2': sentlist[n + 1]})
                    sentid += 1
                except IndexError:
                    newsframe.loc[sentid] = pd.Series({'player': p, 'text': i})
                    sentid += 1

    # concat sentences into one paragraph
    try:
        newsframe['news'] = newsframe[['text', 'text2']].apply(
            lambda x: ''.join(x), axis=1)
    except:
        newsframe['news'] = newsframe['text']

    # final frame
    newframe = newsframe[['player', 'news']]

    def text_sent(row):
        try:
            vs = vaderSentiment(row['news'].encode('utf-8'))
            return(vs['compound'])
        except:
            return(0)

    def generate_id(row):
        concat = row['player'].encode('utf-8') + row['news'].encode('utf-8')
        return(abs(hash(concat)))

    newframe['news_sentiment'] = newframe.apply(text_sent, axis=1)
    newframe['id'] = newframe.apply(generate_id, axis=1)
    newframe['author'] = article['author']
    newframe['crawled'] = article['crawled']
    newframe['published'] = article['published']
    newframe['site'] = article['thread']['site']
    newframe['site_full'] = article['thread']['site_full']
    newframe['site_section'] = article['thread']['site_section']
    newframe['site_type'] = article['thread']['site_type']
    newframe['section_title'] = article['thread']['section_title']
    newframe['title'] = article['thread']['title']
    newframe['title_full'] = article['thread']['title_full']
    newframe['url'] = article['thread']['url']

    return(newframe)


# In[6]:

'''inputs - resp - response object containing articles
          - num - number of response object to get info for

'''


def get_the_sentences(resp, num):

    article = resp.body['posts'][num]
    contentArray = sent_tokenize(article['text'])
    news = parse_article(contentArray, article)
    return(news)


# In[26]:

def get_all_news():
    resp = get_archive()
    finalframe = pd.DataFrame()
    # length = len(resp.body['posts'])
    for z in range(len(resp.body['posts'])):
        try:
            df3 = get_the_sentences(resp, z)
            finalframe = finalframe.append(df3)
        except:
            pass

    #finalframe = finalframe.sort('published', ascending=False)
    return(finalframe)


# In[27]:

def get_vader(row):
    try:
        vader = vaderSentiment(row['news'])['compound']
    except:
	vader = 0
    return(vader)


# inserts df into elastic search
def insert_elastic():

    mapping = {
        "article": {
            "properties": {
                "author": {"type": "string", "index": "not_analyzed"},
                "player": {"type": "string", "index": "analyzed"},
                "crawled": {"type": "date"},
                "news": {"type": "string", "index": "analyzed"},
                "sentiment": {"type": "float"},
                "site": {"type": "string", "index": "not_analyzed"},
                "site_full": {"type": "string", "index": "not_analyzed"},
                "site_section": {"type": "string", "index": "not_analyzed"},
                "site_type": {"type": "string", "index": "not_analyzed"},
                "published": {"type": "date"},
                "section_title": {"type": "string", "index": "not_analyzed"},
                "title": {"type": "string", "index": "not_analyzed"},
                "title_full": {"type": "string", "index": "analyzed"},
                "url": {"type": "string", "index": "not_analyzed"}
            }
        }
    }

    es = Elasticsearch()
    #es.indices.create("mlbnews")
    es.indices.put_mapping(
	index="mlbnews",
        #index="playernews",
        doc_type="article",
        body=mapping)
    df = get_all_news()

    # group by player, take uniques, vader on paragraph, write to csv
    if len(df) > 0:
        df2 = df.groupby(['player'])['news'].apply(lambda x: '\n\t'.join(set(x))).reset_index()
        df2['sentiment'] = df2.apply(get_vader, axis=1)
	df2['chars'] = df2['news'].str.len()
        df2.to_csv('/home/ubuntu/dfsharp/mlb_dfs/webhose_latest.csv', encoding='UTF-8')


    for index, i in df.iterrows():

        # for i in response.body['posts']:
        try:
            es.index(index="mlbnews",#index="playernews",
                     doc_type="article",
                     id=i['id'],
                     op_type="create",
                     body={"author": i['author'],
                           "player": i['player'],
                           "crawled": i['crawled'],
                           "published": i['published'],
                           "news": i['news'],
                           "sentiment": i['news_sentiment'],
                           "site": i['site'],
                           "site_full": i['site_full'],
                           "site_section": i['site_section'],
                           "site_type": i['site_type'],
                           "section_title": i['section_title'],
                           "title": i['title'],
                           "title_full": i['title_full'],
                           "url": i['url']})
        except:
            print('document already exists', i['title'])
            continue


insert_elastic()
