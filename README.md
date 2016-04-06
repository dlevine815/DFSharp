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


*Daily Pipeline*
-generate_model.py
this file runs daily at noon, pulling down stats from rotoguru.net and training a regression model, while splitting off todays players and saving them as a csv

-make_projections.py
this file runs every 10 minutes from noon to 11pm, reading depth charts, injury reports, and latest twitter updates, adjusting projected minutes and then creating projections.  it also inserts the latest projections into elasticsearch.

-optimizer_openopt.py
this is the lineup optimizer file- the optimize() function can take lock or exclude parameters, and a "delta" parameter which changes the optimization function as it relates to ownership.  it also writes the latest lineup to ES.  this file is called from the flask app
