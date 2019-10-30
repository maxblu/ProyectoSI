from flask import Flask
from flask_bootstrap import Bootstrap
from config import Config

app = Flask(__name__,static_url_path='/static', static_folder='static')

Bootstrap(app)
app.config.from_object(Config)
from logic.mainClass import RecuperationEngine

engine = RecuperationEngine()

from app import routes


if  __name__ == "__main__":
    app.run('localhost', port=5000 ,debug=True)
    