# DFSharp
Open Source Tools For Daily Fantasy Basketball

Optimizer and Live projections are at-
DFSharp.com

Kibana dashboard can be viewed at-
DFSharps.com


## *Daily Pipeline*

**-generate_model.py**

this file runs daily at 6am, pulling down stats from rotoguru.net and training/saving a gradient boosted regression model, while splitting off todays players and saving them as a csv

**-make_projections.py**

this file runs every 10 minutes from noon to 11pm, it reads depth charts, injury reports, and live twitter updates, loads latest model, and makes projections.  it inserts the latest projections into elasticsearch and saves a csv of the day's projections

**-optimizer_openopt.py**

this is the lineup optimizer file- the optimize() function can take a host of parameters, and returns a dataframe of the players in the optimized lineup.  it also writes the latest lineup to ES.  this file is called from the flask app

**-/dflask/dflaskr.py**

this is the flask app- it allows users to use the optimizer, manually adjust minutes, and view the latest projections, using AJAX, jquery and datatables.

**-/dflask/templates/layout.html**

this holds the html and javascript required to run the site


### Working files, not yet integrated into dfsharp.com:

-Sentiment.py

this file reads in tweets from this list https://twitter.com/RotoViz/lists/dfs-follows , does sentiment analysis on each one and adds it to elasticsearch

-otp_Elastic.py

this file reads live NBA ownership information from the OwnThePlay.com API and adds it to elasticsearch
[OwnThePlay has shut down until october]

-carrot2.py

this file is for live clustering of elasticsearch results

-webhoser.py

this file uses webhose.io to pull in any articles related to "NBA DFS", and adds them to elasticsearch
