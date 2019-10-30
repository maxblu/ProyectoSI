from flask import Flask
from flask_bootstrap import Bootstrap
from config import Config

app = Flask(__name__,static_url_path='/static', static_folder='static')


Bootstrap(app)
app.config.from_object(Config)

# app.add_url_rule('/')


from app import routes


if  __name__ == "__main__":
    app.run('0.0.0.0', port=5000 ,debug=True)
    