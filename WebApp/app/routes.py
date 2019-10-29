from flask import render_template , url_for , request
from app.forms import SearchTemplate
from app import app
from app import  engine


@app.route('/',methods=['GET', 'POST'])
@app.route('/index',methods=['GET', 'POST'])
def index():
    form = SearchTemplate()
    if request.method == 'POST':
        return search()

    return render_template('index.html', title='Home',form= form)


@app.route('/search',methods=['GET', 'POST'])
def search():
    form = SearchTemplate()
    # engine = CharlotteSearchEngine()
    if form.validate_on_submit():
        search_term = form.query.data # Aqui pasrle la query al sistema para que devuelva las posibles paginas ranqueadas
        is_a_page = 'http://'  in search_term or 'https://'  in search_term
        if is_a_page:
            results , time =   engine.search_query(search_term,weburl=True)
        else:
            results ,time =   engine.search_query(search_term)
            
        return render_template('search_window.html', form=form, results=results,time=time )

    return render_template('search_window.html', title='Search', form=form)