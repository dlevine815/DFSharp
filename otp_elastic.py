import json
from pymongo import MongoClient
import urllib
import time
import numpy as np
from urllib2 import Request, urlopen
from pprint import pprint
from pandas.io.json import json_normalize
from itertools import groupby
import json
import pandas as pd
import ssl
from elasticsearch import Elasticsearch
from datetime import datetime


# InsertElastic- Grabs ownership levels from OwnThePlay.com - inserts into
# elasticsearch
def InsertElastic():

    mapping = {
        "ownership": {
            "properties": {
                "salary": {"type": "integer"},
                "timestamp": {"type": "date"},
                "player": {"type": "string", "index": "not_analyzed"},
                "ownership": {"type": "integer"}
            }
        }
    }

    es = Elasticsearch()
    df = OwnThePlay()
    time = datetime.utcnow()
    for index, row in df.iterrows():
        es.index(index="otp",
                 doc_type="ownership",
                 body={"timestamp": time,
                       "player": row['player'],
                       "salary": row['salary'],
                       "ownership": row['ownership']})


def InsertMongo():

    client = MongoClient('localhost', 27017)
    db = client['otp_db']
    collection = db['ownership_collection']
    records = json.load(urllib.urlopen(
        "http://api.owntheplay.com/api/upcoming/NBA"))
    collection.insert(records)


def OwnThePlay():
    """inputs: none
       outputs: a list of tuples containing ownership percentages
    """

    #context = ssl._create_unverified_context()
    otp_json = json.load(urllib.urlopen(
        "http://api.owntheplay.com/api/upcoming/NBA"))
    # print(otp_json)

    contestSets = otp_json['contestSets']
    league = otp_json['league']  # NBA
    athleteInfo = otp_json['athleteInfo']
    contests = otp_json['contests']
    teams = otp_json['teams']

    today = time.strftime('%Y%m%d')
    todays_games = 0
    # we only want today's contests
    for key in contests:
        contest_date = contests[key]['gameday']
        if contest_date == today:
            todays_games += 1
        # count number of games being played on today's date
        # we will use That number to match with contest set

    print todays_games
    key2 = 0
    for key in contestSets:
        if len(contestSets[key]['contestIds']) == todays_games:
            key2 = key

    # create dataframe of ownership
    ownership_list = []
    yy = 0
    for key in athleteInfo.keys():
        x = athleteInfo[key]['name']
        dic = athleteInfo[key]['contestSets']
        try:
            y = athleteInfo[key]['contestSets'][key2]['salary']
            z = athleteInfo[key]['contestSets'][key2][
                'ownership']['SALARYCAP']['percentage']
            cooltuple = (x, y, z)
            ownership_list.append(cooltuple)
        except KeyError as e:
            # print 'I got a KeyError - reason "%s"' % str(e)
            print 'This player will not be listed: "%s"' % x
        except:
            print 'I got another exception, but I should re-raise'
            raise

    df = pd.DataFrame.from_records(
        ownership_list, columns=[
            'player', 'salary', 'ownership'])
    print df

    return df


InsertElastic()
