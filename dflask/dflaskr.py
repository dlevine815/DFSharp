from __future__ import with_statement
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, redirect, url_for, request, abort, jsonify
import sys
sys.path.append('/home/ubuntu/dfsharp/')

from optimizer_openopt import optimizer
from optimizer_openopt import fuzzy_match
from datetime import datetime, timedelta


import os, cPickle


# get_projections from opt csv
def get_projections():
    # global vars
    today = datetime.today() - timedelta(hours=4)
    yesterday = today - timedelta(hours=24)

    filename = today.strftime('%Y%m%d')+'_opt.csv'
    path = '/home/ubuntu/dfsharp/opt_csvs/'+filename
    # read csv
    try:
	df = pd.read_csv(path)
    except IOError:
	df = pd.read_csv(ypath)
	path = ypath
    df = pd.read_csv(path)
    df[['DK_Proj','min_proj','dk_per_min','value','usage_5g_avg']] = np.round(df[['DK_Proj','min_proj','dk_per_min','value','usage_5g_avg']],1)
    ajax = df[['numpos','name','Team','Opp','dk_sal','ownership','DK_Proj','dvprank','min_proj','dk_per_min','value','usage_5g_avg']].to_json(orient='records')
    print(ajax)
    return(ajax)



# adjust_minutes
# reads in file from generate_model
# adjusts min_proj for a specific player
def adjust_minutes(player, minutes):
    filename = today.strftime('%Y%m%d')+'_players.csv'
    path = '/home/ubuntu/dfsharp/csvs/'+filename
    df = pd.read_csv(path)
    # fuzzy match requested player
    playernames = df['name'].tolist()
    playername = fuzzy_match(player, playernames)
    df.loc[df.name == playername, 'min_3g_avg'] = float(minutes)
    df.to_csv(path)
    return([player, minutes])
    
    


def run_in_separate_process(func, *args, **kwds):
    pread, pwrite = os.pipe()
    pid = os.fork()
    if pid > 0:
        os.close(pwrite)
        with os.fdopen(pread, 'rb') as f:
            status, result = cPickle.load(f)
        os.waitpid(pid, 0)
        if status == 0:
            return result
        else:
            raise result
    else: 
        os.close(pread)
        try:
            result = func(*args, **kwds)
            status = 0
        except Exception, exc:
            result = exc
            status = 1
        with os.fdopen(pwrite, 'wb') as f:
            try:
                cPickle.dump((status,result), f, cPickle.HIGHEST_PROTOCOL)
            except cPickle.PicklingError, exc:
                cPickle.dump((2,exc), f, cPickle.HIGHEST_PROTOCOL)
        os._exit(0)


app = Flask(__name__)

# feeds latest projections into site
@app.route('/v')
def pre3():
    # send projections in datatable format to webpage
    data = get_projections()
    return data


# minutes adjustment api- takes input from field, adjusts minutes in file
@app.route('/u')
def pre2():
    # get locks field
    tx = request.args.get('uu', 0, type=str)
    if len(tx) > 0:
	splist = tx.split(", ")
        player = splist[0]
	minutes = splist[1]
	data = adjust_minutes(player, minutes)
	print('ADJUSTED************************************: '+player+minutes)
    	return jsonify(result=data)
    else:
	return jsonify(result=['No adjustment made'])


@app.route('/t')
def pre():
    # get locks field
    tx = request.args.get('aa', 0, type=str)
    locks = []
    if len(tx) > 0:
	locks = tx.split(", ")
    # get excludes
    ex = request.args.get('bb', 0, type=str)
    excludes = []
    if len(ex) > 0:
	excludes = ex.split(", ")
    # get min ownership
    min_own = 0
    mo = request.args.get('cc', 0, type=int)
    if mo > 0:
	min_own = mo
    # get max ownership
    max_own = 100
    maxo = request.args.get('dd', 0, type=int)
    if maxo <100 and maxo >0:
	max_own = maxo
    # get min dvp
    min_dvp = 0
    md = request.args.get('ee', 0, type=int)
    if md > 0 and md <= 5:
	min_dvp = md
    # get min salary
    min_sal = 2000
    ms = request.args.get('ff', 0, type=int)
    if ms > 2000 and ms < 10000:
	min_sal = ms
    # get delta
    delta = 5
    ds = request.args.get('gg', 0, type=int)
    if ds < 5 and ds >= -5:
	delta = ds
   
    data = run_in_separate_process(optimizer, locks=locks, exclusions=excludes, delta=int(delta), min_own=min_own, min_dvp=min_dvp, min_sal=min_sal, max_own=max_own)
    return(data)

@app.route('/', methods=['GET', 'POST'])
def home_page():
    locks = []
    excludes = []
    if request.method == 'POST':
	print "HOME PAGE POST"
    else:
        print "HOME PAGE GET"
        return render_template('layout.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
