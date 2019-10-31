from flask import render_template , url_for , request , send_file , redirect 
from flask_paginate import Pagination , get_page_parameter , get_page_args
from app.forms import SearchTemplate,SelectDirectory, SaveFeedBack
from app import app
from tkinter import filedialog
import os
from logic.mainClass import RecuperationEngine


global BASE_DIR
global engine
global results
global time
global search_query

@app.route('/',methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    form = SelectDirectory()
    if request.method == 'POST':

        global engine
        global BASE_DIR
        global results
        global search_query
        results = []

        BASE_DIR = filedialog.askdirectory()
        engine = RecuperationEngine()
        engine.load_folder(BASE_DIR)
        engine.save_tfidf_matrix()
        search_query = []
        return redirect('/search')

    return render_template('index.html', title='Home',form= form)




@app.route('/search',methods=['GET', 'POST'])
def search():
    global engine
    global results
    global time
    global search_query
    form = SearchTemplate()
    relevantForm =  SaveFeedBack()

    if not len(request.form.getlist('relevant')) == 0:
        engine.add_retro_feed(search_query,request.form.getlist('relevant'))
        engine.save_retro_feed()

    if form.validate_on_submit() :
        print(engine.retro_feed_data)        

        search_term = form.query.data # Aqui pasrle la query al sistema para que devuelva las posibles paginas ranqueadas
        search_query = search_term

        is_a_page = 'http://'  in search_term or 'https://'  in search_term
        if is_a_page:
            results , time =   engine.search_query(search_term,weburl=True)
        else:
            results ,time =    engine.search_query(search_term)
        
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
    global BASE_DIR
    print("entre")

    abs_path = os.path.join(BASE_DIR,req_path)
    return send_file(abs_path)
    