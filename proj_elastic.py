#!/bin/env python
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch


#df = pd.read_csv('/home/ubuntu/dfsharp/opt_csvs/'+date.strftime('%Y%m%d')+'_opt.csv')

def InsertProj(df, indexer="projections"):

    mapping = {
        "dataframe": {
            "properties": {
                #"GameID": {"type": "string", "index": "not_analyzed"},
                #"gametime" : {"type" :
                "numpos": {"type": "string"},
                "player": {"type": "string", "index": "not_analyzed"},
                "dk_sal": {"type": "float"},
                "Team": {"type": "string"},
                "Start": {"type": "boolean"},
                "DK_Proj": {"type": "float"},
                "value": {"type": "float"},
                "ceiling": {"type": "float"},
                "min_proj": {"type": "float"},
                "dk_per_min": {"type": "float"},
                "home": {"type": "string"},
                "status": {"type": "string", "index": "not_analyzed"},
                "timestamp": {"type": "date"},
                "ownership": {"type": "float"},
                "Opp": {"type": "string"},
                "dvp": {"type": "float"},
                "dvprank": {"type": "float"},
                "otp_value": {"type": "float"},

                "kpos": {"type": "float"},
                "starts_past_week": {"type": "float"},
                "b2b": {"type": "boolean"},
                "usage_3g_avg": {"type": "float"},
                "usage_5g_avg": {"type": "float"},
                "value_3g_avg": {"type": "float"},
                "mvs_5g_avg": {"type": "float"},
                "starter_5g_avg": {"type": "float"},
                "proj_pure": {"type": "float"}
            }
        }
    }

    es = Elasticsearch()
    es.indices.delete(index=indexer, ignore=[400, 404])
    es.indices.create(indexer, ignore=400)
    es.indices.put_mapping(index=indexer, doc_type="dataframe", body=mapping)
    date = datetime.today()
    time = datetime.utcnow()

    for index, row in df.iterrows():
        es.index(index=indexer,
                 doc_type="dataframe",
                 body={"numpos": row['numpos'],
                          "player": row['name'],
                          "dk_sal": row['dk_sal'],
                          "Team": row['Team'],
                          "Start": row['Start'],
                          "DK_Proj": row['DK_Proj'],
                          "value": row['value'],
                          "ceiling": row['ceiling'],
                          "min_proj": row['min_proj'],
                          "dk_per_min": row['dk_per_min'],
                          "home": row['home'],
                          "status": row['status'],
                          "ownership": row['ownership'],
                          "Opp": row['Opp'],
                          "dvp": row['dvp'],
                          "dvprank": row['dvprank'],
                          "otp_value": row['otp_value'],

                       "kpos": row['kpos'],
                       "starts_past_week": row['starts_past_week'],
                       "b2b": row['b2b'],
                       "usage_3g_avg": row['usage_3g_avg'],
                       "usage_5g_avg": row['usage_5g_avg'],
                       "value_3g_avg": row['value_3g_avg'],
                       "mvs_5g_avg": row['mvs_5g_avg'],
                       "starter_5g_avg": row['starter_5g_avg'],
                       "proj_pure": row['proj_pure'],


                          "timestamp": time})


# def InsertOptimal():


# InsertProj()


def InsertLogs(df, indexer="gamelogs"):

    mapping = {
        "dataframe": {
            "properties": {
                "timestamp": {"type": "date"},
                "GameID": {"type": "string", "index": "not_analyzed"},
                "gametime": {"type": "float"},
                "player": {"type": "string", "index": "not_analyzed"},
                "Minutes": {"type": "float"},
                "Start": {"type": "boolean"},
                "active": {"type": "boolean"},
                "DKP": {"type": "float"},
                "Team": {"type": "string", "index": "not_analyzed"},
                "Opp": {"type": "string", "index": "not_analyzed"},
                "home": {"type": "boolean"},
                "team_pts": {"type": "float"},
                "opp_pts": {"type": "float"},
                "dk_sal": {"type": "float"},
                "dk_pos": {"type": "float"},
                "dk_change": {"type": "float"},
                "kpos": {"type": "float"},
                "min_90d_avg": {"type": "float"},
                "dk_avg_90_days": {"type": "float"},
                "dk_per_min": {"type": "float"},
                "dk_std_90_days": {"type": "float"},
                "dk_max_30_days": {"type": "float"},
                "min_when_start": {"type": "float"},
                "min_when_bench": {"type": "float"},
                "starts_past_week": {"type": "float"},
                "min_proj": {"type": "float"},
                "dvp": {"type": "float"},
                "dvprank": {"type": "float"},
                "pts": {"type": "float"},
                "rbs": {"type": "float"},
                "stl": {"type": "float"},
                "ast": {"type": "float"},
                "blk": {"type": "float"},
                "3pm": {"type": "float"},
                "fgm": {"type": "float"},
                "fga": {"type": "float"},
                "ftm": {"type": "float"},
                "fta": {"type": "float"},
                "tov": {"type": "float"},
                "min_yest": {"type": "float"},
                "b2b": {"type": "boolean"},
                "usage": {"type": "float"},
                "gp": {"type": "float"},
                "min_3g_avg": {"type": "float"},
                "usage_3g_avg": {"type": "float"},
                "usage_5g_avg": {"type": "float"},
                "value_3g_avg": {"type": "float"},
                "starter_min": {"type": "float"},
                "min_vs_starters": {"type": "float"},
                "mvs_5g_avg": {"type": "float"},
                "starter_5g_avg": {"type": "float"},
                "value": {"type": "float"}
            }
        }
    }

    es = Elasticsearch()
    es.indices.delete(index=indexer, ignore=[400, 404])
    es.indices.create(indexer, ignore=400)
    es.indices.put_mapping(index=indexer, doc_type="dataframe", body=mapping)
    date = datetime.today()
    time = datetime.utcnow()

    def addval(row):
        if row['DKP'] > 0:
            val = row['DKP'] / (row['dk_sal'] / 1000)
            return(val)
        else:
            return(0)

    df['value'] = df.apply(addval, axis=1)
    df = df.fillna(value=0)

    for index, row in df.iterrows():
        es.index(index=indexer,
                 id=hash(str(row['GameID']) + str(row['name'])),
                 doc_type="dataframe",
                 body={"timestamp": row['index'],
                          "GameID": row['GameID'],
                          "gametime": row['gametime'],
                          "player": row['name'],
                          "Minutes": row['Minutes'],
                          "Start": row['Start'],
                          "active": row['active'],
                          "DKP": row['DKP'],
                          "Team": row['Team'],
                          "Opp": row['Opp'],
                          "home": row['home'],
                          "team_pts": row['team_pts'],
                          "opp_pts": row['opp_pts'],
                          "dk_sal": row['dk_sal'],
                          "dk_pos": row['dk_pos'],
                          "dk_change": row['dk_change'],
                          "kpos": row['kpos'],
                          "min_90d_avg": row['min_90d_avg'],
                          "dk_avg_90_days": row['dk_avg_90_days'],
                          "dk_per_min": row['dk_per_min'],
                          "dk_std_90_days": row['dk_std_90_days'],
                          "dk_max_30_days": row['dk_max_30_days'],
                          "min_when_start": row['min_when_start'],
                          "min_when_bench": row['min_when_bench'],
                          "starts_past_week": row['starts_past_week'],
                          "min_proj": row['min_proj'],
                          "dvp": row['dvp'],
                          "dvprank": row['dvprank'],
                          "pts": row['pts'],
                          "rbs": row['rbs'],
                          "stl": row['stl'],
                          "ast": row['ast'],
                          "blk": row['blk'],
                          "3pm": row['3pm'],
                          "fgm": row['fgm'],
                          "fga": row['fga'],
                          "ftm": row['ftm'],
                          "fta": row['fta'],
                          "tov": row['tov'],
                       "min_yest": row['min_yest'],
                       "b2b": row['b2b'],
                       "usage": row['usage'],
                       "gp": row['gp'],
                       "min_3g_avg": row['min_3g_avg'],
                       "usage_3g_avg": row['usage_3g_avg'],
                       "usage_5g_avg": row['usage_5g_avg'],
                       "value_3g_avg": row['value_3g_avg'],
                       "starter_min": row['starter_min'],
                       "min_vs_starters": row['min_vs_starters'],
                       "mvs_5g_avg": row['mvs_5g_avg'],
                       "starter_5g_avg": row['starter_5g_avg'],
                          "value": row['value']})
