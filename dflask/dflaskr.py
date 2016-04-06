import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, redirect, url_for, request, abort
import sys
sys.path.append('/home/ubuntu/dfsharp/')

from optimizer_openopt import optimizer


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
	    result = optimizer(locks=locks, exclusions=excludes, delta=int(delta))
            return render_template('layout.html', data=result)
	    #return render_template('layout.html', data=["<--- Your lineup!"])
	except:
	    return render_template('layout.html', data=["Sorry, no lineup found"])
    else:
        print "HOME PAGE GET"
        return render_template('layout.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
