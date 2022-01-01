from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired

class InputOutputTextForm(FlaskForm):
    input_text = TextAreaField('input_text', validators=[DataRequired()])
    submit = SubmitField("submit")

    output_text = TextAreaField("output_text",
                                render_kw={'readonly': True})