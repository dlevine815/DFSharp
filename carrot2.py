from elasticsearch import Elasticsearch
from pprint import pprint


es = Elasticsearch()

result = es.transport.perform_request('POST', '/webhose/_search_with_clusters',
	body = { "search_request": {
					"fields" : ["text" ],
		"query" : {
			"bool" : {
				"must":  [{
					#"match_all" : {}
					"match" : {
						"text" : "Value"
						  }
					  },
					{
					"range" : {
						"timestamp" : {
							"gte" : "now-3d"
					       	      	      }
					  	  }
				       	}]
				  }
			  },
				"size": 100
				     },
				"query_hint": "Value",
				"field_mapping": {
						"title": ["fields.text" ]
						},
				 "algorithm" : "stc" 
		 }  )

pprint(len(result[1]['clusters']))
numclusts = len(result[1]['clusters'])

#pprint(result[1]['clusters'])

for i in range(numclusts):
    pprint(result[1]['clusters'][i]['label'])
    score = result[1]['clusters'][i]['score']
    docs = result[1]['clusters'][i]['documents']
    print(len(docs), score)

#pprint(result['clusters'])
#pprint(result[0]['200'])
