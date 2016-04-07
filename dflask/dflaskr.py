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
	try:
	    result = run_in_separate_process(optimizer, locks=locks, exclusions=excludes, delta=int(delta))
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
