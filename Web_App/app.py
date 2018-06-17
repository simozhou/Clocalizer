from flask import Flask, render_template, request
import keras.models
import re
import os
import sys
import input_soap as inp
import numpy as np
import subprocess as sp

sys.path.append(os.path.abspath("./model"))

from loader import *

application = Flask(__name__)

global model, graph

model, graph = init()


def clean_sequence(sequence):
    return re.sub(r'[\s\n]', '', sequence)


@application.route('/')
def index():
    return render_template('index.html', prediction="None")


@application.route('/predict/', methods=['GET', 'POST'])
def predict():
    classes = ['Nucleus', 'Cytoplasm', 'Extracellular', 'Mitochondrion', 'Cell membrane', 'ER',
               'Chloroplast', 'Golgi apparatus', 'Lysosome', 'Vacuole']
    sequence = clean_sequence(request.form['sequence'])
    # pssm = inp.make_input(sequence)
    pssm, err = inp.psiblaster(sequence)
    pssm = inp.ohe_tailor(pssm, 1000)
    feedable = inp.reshaper(pssm, 'cnn')
    with graph.as_default():
        result = classes[np.argmax(model.predict(feedable))]

    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    application.run(host='0.0.0.0')
