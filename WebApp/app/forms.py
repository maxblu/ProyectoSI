from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class SearchTemplate(FlaskForm):
    query = StringField('Search', validators=[DataRequired()])
    submit = SubmitField('Look For It!')

class SelectDirectory(FlaskForm):
    lsi = BooleanField('LSI Model')
    vectorial =  BooleanField('Vectorial Model')
    submit = SubmitField('Select Directory!')

class SaveFeedBack(FlaskForm):
    submit = SubmitField('save relevant documents')