from elasticsearch import Elasticsearch
from datetime import datetime
from pprint import pprint


es = Elasticsearch()

result = es.transport.perform_request('POST', '/nlp2/_search_with_clusters',
                                      body={"search_request": {
                                          "fields": ["text"],
                                          "query": {
                                              "bool": {
                                                  "must": [{
                                                      #"match_all" : {}
                                                      "match": {
                                                          "text": "Injury, OUT, QUESTIONABLE, DOUBTFUL, PROBABLE, START, STARTING, BENCH"
                                                          #"text" : "Injury"
                                                          #"text" : "Jimmy Butler"
                                                      }
                                                  },
                                                      {
                                                      "range": {
                                                          "timestamp": {
                                                              "gte": "now-4h"
                                                          }
                                                      }
                                                  }]
                                              }
                                          },
                                          "size": 100
                                      },
                                          "query_hint": "",
                                          "field_mapping": {
                                          "title": ["fields.text"]
                                      },
                                          "algorithm": "stc"
                                      })

# pprint(result[1])
# pprint(type(result[1]))
# for k, v in result[1].v():
# for key in result[1].iteritems():
# print(key[0])


# organize mapping for new elasticsearch index
mapping = {
    "clustered_tweet": {
        "properties": {
            "timestamp": {"type": "date"},
            "text": {"type": "string", "index": "analyzed"}
        }
    }
}

# create index for clusterhits
# es.indices.create("clusterhits")
es.indices.put_mapping(
    index="clusterhits",
    doc_type="clustered_tweet",
    body=mapping)
# get dict of clustered tweets
hitdict = result[1]['hits']

for i in hitdict['hits']:
    #twitid = hash(str(i['_score'])+str(i['fields']['text'][0].encode('ascii','ignore')))
    twitid = hash(str(i['fields']['text'][0].encode('ascii', 'ignore')))
    time = datetime.utcnow()
    print(i['fields']['text'][0])
    try:
        es.index(index="clusterhits",
                 doc_type="clustered_tweet",
                 id=twitid,
                 op_type="create",
                 body={"timestamp": time,
                       "text": i['fields']['text'][0]})
    except:
        continue
        #"player": {"type": "string"


# for key in hitdict.iteritems():
#    print(key[0])
# for i in range(len(hits)):
    # print(hits[i])
# print(hitdict['hits'][0])


numclusts = len(result[1]['clusters'])

# pprint(result[1]['clusters'])
for i in range(numclusts):
    pprint(result[1]['clusters'][i]['label'])
    score = result[1]['clusters'][i]['score']
    docs = result[1]['clusters'][i]['documents']
    print(len(docs), score)

# pprint(result['clusters'])
# pprint(result[0]['200'])
