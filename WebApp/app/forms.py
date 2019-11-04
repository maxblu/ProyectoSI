from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, IntegerField
from wtforms.validators import DataRequired

class SearchTemplate(FlaskForm):
    query = StringField('Search', validators=[DataRequired()])
    submit = SubmitField('Look For It!')

class SelectDirectory(FlaskForm):
    lsi = BooleanField('LSI Model')
    vectorial =  BooleanField('Vectorial Model')
    k = IntegerField('K-LSI')
    rank_cant = IntegerField('Rank')
    submit = SubmitField('Select Directory!')

class SaveFeedBack(FlaskForm):
    submit = SubmitField('save relevant documents')