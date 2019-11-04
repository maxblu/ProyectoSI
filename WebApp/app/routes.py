from flask import render_template , url_for , request , send_file , redirect 
from flask_paginate import Pagination , get_page_parameter , get_page_args
from app.forms import SearchTemplate,SelectDirectory, SaveFeedBack
from app import app
from logic.medidas import *
from tkinter import filedialog
import tkinter as tk
import os
import json
from logic.mainClass import RecuperationEngine


global BASE_DIR
global engine
global results
global resultsi
global time
global timei
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
        global resultsi
        global search_query
        global model
        results = []
        resultsi = []
        lsi =form.lsi.data
        vec = form.vectorial.data

        BASE_DIR = filedialog.askdirectory()

        if vec and lsi:
            model= 'both'
            engine = RecuperationEngine(model= model, BASE_DIR = BASE_DIR )
        elif lsi:
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
        # else:
        #     model= 'lsi-gen'
        #     engine = RecuperationEngine(model= model, BASE_DIR = BASE_DIR )
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
    global timei
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

        print('term:', search_term)
        results ,time = engine.search_query(search_term, model= model)
        
        try:
            metrics = engine.retro_feed_data[search_query][-1]
        except:
            metrics = [0,0,0,0,0]        

        page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
        total = len(results)


        subresult = results[0:per_page]
        paginator = Pagination(total,page =page , total= total,per_page = per_page, search=True ,css_framework='bootstrap4' , record_name='results' )

        return render_template('search_window.html',relevantForm= relevantForm, form=form, paginator=paginator, results=subresult,time=time,metrics = metrics )

    total = len(results)
    if request.args.get('page') or not total == 0:
        page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')


        paginator = Pagination(total,page =page , total= total, per_page = per_page, search=True ,css_framework='bootstrap4' , record_name='results' )
        subresult= results[offset: offset+per_page]

        return render_template('search_window.html',relevantForm=relevantForm, form=form, paginator=paginator, results= subresult,time=time ,metrics = metrics)

    return render_template('search_window.html', title='Search',  form=form)

@app.route('/relevants',methods=['GET', 'POST'])
def relevant():
    global engine
    global search_query

    form = SearchTemplate()
    relevantForm =  SaveFeedBack()

    results = engine.file_names
    total = len(results)

    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    paginator = Pagination(total, page=page, total=total, per_page=per_page, search=True, css_framework='bootstrap4' , record_name='results' )
    subresult = zip(results[offset: offset+per_page],[t[:min(100,len(t))]+"..." for t in engine.docs_prepoced[offset: offset+per_page]])

    if form.validate_on_submit() :
        search_query = form.query.data 
        if not len(request.form.getlist('relevant')) == 0:
            engine.add_relevant(search_query, request.form.getlist('relevant'))
            engine.save_relevante()
        return render_template('relevants.html',relevantForm=relevantForm, form=form, paginator=paginator, results=subresult )

    total = len(results)
    if request.args.get('page') or not total == 0:
        return render_template('relevants.html',relevantForm=relevantForm, form=form, paginator=paginator, results= subresult)

    return render_template('relevants.html', title='relevant',  form=form)

@app.route('/compare',methods=['GET', 'POST'])
def compare():
    global engine
    global results
    global resultsi
    global time
    global timei
    global search_query
    # global model
    global metrics
                
    active = engine.model == "both"
    
    form = SearchTemplate()
    if form.validate_on_submit() :

        search_term = form.query.data # Aqui pasrle la query al sistema para que devuelva las posibles paginas ranqueadas
        search_query = search_term

        results,    time = engine.search_query(search_term, model= 'vec')
        resultsi,   timei = engine.search_query(search_term, model= 'lsi-gensim')

        page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
        total = len(resultsi)

        subresult  = results[0:per_page]
        subresulti = resultsi[0:per_page]

        zi = zip(subresult, subresulti)

        paginator  = Pagination(total,page =page , total=total, per_page = per_page, search=True , css_framework='bootstrap4' , record_name='results' )
        return render_template('compare.html', form=form, paginator=paginator, results=zi,timei = timei, time=time,active = active)

    total = len(resultsi)
    if request.args.get('page') or not total == 0:
        page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')

        paginator = Pagination(total,page =page , total= total, per_page = per_page, search=True ,css_framework='bootstrap4' , record_name='results' )

        subresult   = results[offset: offset+per_page]
        subresulti  = resultsi[offset: offset+per_page]

        zi = zip(subresult, subresulti)

        return render_template('compare.html', form=form, paginator=paginator, results= zi,timei=timei,time=time , active= active)

    return render_template('compare.html', title='Compare',  form=form, active = active)

@app.route('/test_case_results')
def show_test_case_results():
    with open("data/corpus1_full_stats.json", encoding='utf-8') as fd:
        d = json.load(fd)

    a = [0,0,0,0,0]
    b = [0,0,0,0,0]
    for key in d:
        a = [(x+y) for x,y in zip(a, d[key]["measures-vec"])]
        b = [(x+y) for x,y in zip(b, d[key]["measures-lsi"])]

    a = [round(x / len(d),3) for x in a]
    b = [round(x / len(d),3) for x in b]

    return render_template('test_case_results.html', measures=d, proma=a, promb=b)

@app.route('/', defaults={'req_path':''} )
@app.route('/<path:req_path>')
def data(req_path):
    if req_path == 'favicon.ico':
        return send_file('static/favicon.ico')

    global BASE_DIR


    abs_path = os.path.join(BASE_DIR,req_path)
    return send_file(abs_path)
