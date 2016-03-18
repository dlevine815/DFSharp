import unirest
import time
import webhose
from elasticsearch import Elasticsearch
import json
import cnfg

config = cnfg.load("/home/ubuntu/dfsharp/.webhoser_config")
tok = config["token"]
webhose.config(token=tok)



mapping = {
	"article": {
		"properties": {
			"author": {"type": "string", "index": "not_analyzed"},
			"crawled": {"type": "date"},
			"language": {"type": "string"},
			"text": {"type": "string", "index": "analyzed"},
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

#es = Elasticsearch()
#es.indices.create("webhose")
#es.indices.put_mapping(index="webhose", doc_type="article", body=mapping)

def hose_stream():
    es = Elasticsearch()

    r = webhose.search("NBA DFS")
    js = r.response.json()
    while True:
	j = 0
        for post in r:
	    # print(post.thread.title)
	    try:
                es.index(index="webhose",
		 doc_type="article",
		id=js['posts'][j]['uuid'],
		op_type="create",
		 body={ "author" : post.author,
			"crawled": post.crawled,
			"language": post.language,
			"published": post.published,
			"text": post.text,
			"site": post.thread.site,
			"site_full": post.thread.site_full,
			"site_section": post.thread.site_section,
			"site_type": post.thread.site_type,
			"section_title": post.thread.section_title,
			"title": post.thread.title,
			"title_full": post.thread.title_full,
			"url": post.thread.url})
		j += 1	
	    except:
		j += 1
	        # print('document already exists')
		continue
	time.sleep(1800)
	r = r.get_next()






def get_archive():
    # get response from webhose query "NBA DFS"
    # response = unirest.get("https://webhose.io/search?token="+tok+"&format=json&q=NBA%20DFS&ts=1455442050890",
    # response = unirest.get("https://webhose.io/search?token="+tok+"&format=json&ts=1456157125174&q=NBA+DFS",
    # response = unirest.get("https://webhose.io/search?token="+tok+"&format=json&ts=1456503719322&q=NBA+DFS",
    # response = unirest.get("https://webhose.io/search?token="+tok+"&format=json&ts=1456897517907&q=NBA+DFS",
    # response = unirest.get("https://webhose.io/search?token="+tok+"&format=json&ts=1457242057608&q=NBA+DFS",
    # response = unirest.get("https://webhose.io/search?token="+tok+"&format=json&ts=1457589738327&q=NBA+DFS",
    response = unirest.get("https://webhose.io/search?token="+tok+"&format=json&q=NBA+DFS",

    headers={
    "Accept": "application/json"
    }
    )

    es = Elasticsearch()
    print(len(response.body['posts']))
    

    for i in response.body['posts']:
	try:
            es.index(index="webhose",
		 doc_type="article",
		 id=i['uuid'],
		 op_type="create",
		 body={ "author" : i['author'],
			"crawled": i['crawled'],
			"language": i['language'],
			"published": i['published'],
			"text": i['text'],
			"site": i['thread']['site'],
			"site_full": i['thread']['site_full'],
			"site_section": i['thread']['site_section'],
			"site_type": i['thread']['site_type'],
			"section_title": i['thread']['section_title'],
			"title": i['thread']['title'],
			"title_full": i['thread']['title_full'],
			"url": i['thread']['url']})
	except:
	    print('document already exists', i['thread']['title'])
	    continue

	
get_archive()
# hose_stream()
