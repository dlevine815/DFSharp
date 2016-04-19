from __future__ import with_statement
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, redirect, url_for, request, abort
import sys
sys.path.append('/home/ubuntu/dfsharp/')

from optimizer_openopt import optimizer


import os, cPickle


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

@app.route('/', methods=['GET', 'POST'])
def home_page():
    locks = []
    excludes = []
    if request.method == 'POST':
	print "HOME PAGE POST"
        if request.form['locks'] and request.form['excludes']:
            lockstring = request.form['locks']
	    locks = lockstring.split(", ")
            exstring = request.form['excludes']
	    excludes = exstring.split(", ")
        elif request.form['locks']:
            locks = request.form['locks'].split(", ")
        elif request.form['excludes']:
            excludes = request.form['excludes'].split(", ")

	delta = request.form['slider']
	if request.form['min_ownership']:
	    min_own = request.form['min_ownership']
	else:
	    min_own = 0
	if request.form['min_dvp']:
	    min_dvp = request.form['min_dvp']
	else:
	    min_dvp = 0
	if request.form['max_own']:
	    max_sal = request.form['max_own']
	else:
	    max_own = 100
	if request.form['min_sal']:
	    min_sal = request.form['min_sal']
	else:
	    min_sal= 3000
	try:
	    result = run_in_separate_process(optimizer, locks=locks, exclusions=excludes, delta=int(delta), min_own=min_own, min_dvp=min_dvp, min_sal=min_sal, max_own=max_own)
	    #result = optimizer(locks=locks, exclusions=excludes, delta=int(delta))
            return render_template('layout.html', data=result)
	    #return render_template('layout.html', data=["<--- Your lineup!"])
	except:
	    return render_template('layout.html', data=["Sorry, no lineup found"])
    else:
        print "HOME PAGE GET"
        return render_template('layout.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
