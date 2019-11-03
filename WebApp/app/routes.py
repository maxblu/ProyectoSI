from flask import render_template , url_for , request , send_file , redirect 
from flask_paginate import Pagination , get_page_parameter , get_page_args
from app.forms import SearchTemplate,SelectDirectory, SaveFeedBack
from app import app
from logic.medidas import *
from tkinter import filedialog
import tkinter as tk
import os
from logic.mainClass import RecuperationEngine


global BASE_DIR
global engine
global results
global time
global search_query
global model
global metrics

@app.route('/',methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    form = SelectDirectory()
    if request.method == 'POST':

        global engine
        global BASE_DIR
        global results
        global search_query
        global model
        results = []
        lsi =form.lsi.data
        vec = form.vectorial.data

        BASE_DIR = filedialog.askdirectory()

        if lsi:
            model= 'lsi-gensim'
            # engine.save_tfidf_matrix()
            # engine.LSA()
            engine = RecuperationEngine(model= model, BASE_DIR = BASE_DIR )
            # engine.save_lsi_gsim()
        elif vec:
            model= 'vec'
            engine = RecuperationEngine(model= model, BASE_DIR = BASE_DIR )
            # engine.save_tfidf_matrix()
            # engine.LSA()
        else:
            model= 'lsi-gen'
            engine = RecuperationEngine(model= model, BASE_DIR = BASE_DIR )
            # engine.save_tfidf_matrix()
            # engine.LSA()

        

        # engine.load_folder(BASE_DIR)

        search_query = []
        return redirect('/search')

    return render_template('index.html', title='Home',form= form)

@app.route('/statistics')
def show_statistics():
    global engine
    print("entro")
    return render_template('statistic.html',measures= engine.retro_feed_data)


@app.route('/search',methods=['GET', 'POST'])
def search():
    global engine
    global results
    global time
    global search_query
    global model
    global metrics

    form = SearchTemplate()
    relevantForm =  SaveFeedBack()

    if not len(request.form.getlist('relevant')) == 0:
        
        engine.add_retro_feed(search_query,request.form.getlist('relevant'),results)
        engine.save_retro_feed()

    if form.validate_on_submit() :

        search_term = form.query.data # Aqui pasrle la query al sistema para que devuelva las posibles paginas ranqueadas
        search_query = search_term

        results ,time ,precc, recb, f_med, f1_med, r_prec = engine.search_query(search_term, model= model)

        

        page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
        total = len(results)


        subresult = results[0:per_page +1]
        paginator = Pagination(total,page =page , total= total,per_page = per_page, search=True ,css_framework='bootstrap4' , record_name='results' )

        return render_template('search_window.html',relevantForm= relevantForm, form=form, paginator=paginator, results=subresult,time=time )

    total = len(results)
    if request.args.get('page') or not total == 0:
        page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')


        paginator = Pagination(total,page =page , total= total, per_page = per_page, search=True ,css_framework='bootstrap4' , record_name='results' )
        subresult= results[offset: offset+per_page]

        return render_template('search_window.html',relevantForm=relevantForm, form=form, paginator=paginator, results= subresult,time=time )

    return render_template('search_window.html', title='Search',  form=form)

@app.route('/', defaults={'req_path':''} )
@app.route('/<path:req_path>')
def data(req_path):
    if req_path == 'favicon.ico':
        return send_file('static/favicon.ico')

    global BASE_DIR


    abs_path = os.path.join(BASE_DIR,req_path)
    return send_file(abs_path)
