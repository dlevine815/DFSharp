# DFSharp
Open Source Tools For Daily Fantasy Basketball

Kibana dashboard can be viewed at-
DFSharps.com


-Sentiment.py
this file reads in tweets from this list https://twitter.com/RotoViz/lists/dfs-follows , does sentiment analysis on each one and adds it to elasticsearch

-otp_Elastic.py
this file reads live NBA ownership information from the OwnThePlay.com API and adds it to elasticsearch

-carrot2.py
this file is for live clustering of elasticsearch results

-webhoser.py
this file uses webhose.io to pull in any articles related to "NBA DFS", and adds them to elasticsearch
