import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch

def InsertProj():

    mapping = {
	"dataframe": {
		"properties": {
			#"GameID": {"type": "string", "index": "not_analyzed"},
			#"gametime" : {"type" : 
			"numpos" : {"type": "string"},
			"name" : {"type": "string", "index": "not_analyzed"},
			"dk_sal" : {"type": "float"},
			"Team" : {"type": "string"},
			"Start" : {"type": "boolean"},
			"DK_Proj": {"type": "float"},
			"value" : {"type": "float"},
			"ceiling": {"type": "float"},
			"min_proj": {"type": "float"},
			"dk_per_min": {"type": "float"},
			"home": {"type": "string"},
			"opppts_avg": {"type": "float"},
			"status" : {"type": "string", "index": "not_analyzed"},
			"timestamp": {"type": "date"}
				}
			}
		}

    es = Elasticsearch()
    es.indices.delete(index="projections", ignore=[400, 404])
    es.indices.create("projections", ignore=400)
    es.indices.put_mapping(index="projections", doc_type="dataframe", body=mapping)
    date = datetime.today()
    time = datetime.utcnow()
    df = pd.read_csv('/home/ubuntu/dfsharp/opt_csvs/'+date.strftime('%Y%m%d')+'_opt.csv')

    for index, row in df.iterrows():
	es.index(index="projections",
		 doc_type="dataframe",
		 body = { "numpos" : row['numpos'],
			  "name" : row['name'],
			  "dk_sal" : row['dk_sal'],
			  "Team" : row['Team'],
			  "Start" : row['Start'],
			  "DK_Proj" : row['DK_Proj'],
			  "value" : row['value'],
			  "ceiling" : row['ceiling'],
			  "min_proj" : row['min_proj'],
			  "dk_per_min" : row['dk_per_min'],
			  "home" : row['home'],
			  "opppts_avg" : row['opppts_avg'],
			  "status" : row['status'],
			  "timestamp": time})
		


InsertProj()







