from flask import render_template

from .app import app
from .forms import InputOutputTextForm
from .model import get_output_text

@app.route('/', methods=["GET", "POST"])
def main():
    template = "index.html"
    form = InputOutputTextForm()

    if form.validate_on_submit():
        form.output_text.data = get_output_text(form.input_text.data)

    return render_template(template, form=form)
