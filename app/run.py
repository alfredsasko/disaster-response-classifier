import sys
import json
import plotly
import numpy as np
import pandas as pd
from scipy import sparse as sp

from flask import Flask
from flask import render_template, request, jsonify
from sqlalchemy import create_engine

# enable access to feature_extracton module during model unpickling
sys.path.append(r'../models/')

# enable access to models package
sys.path.append(r'../')

from data.process_data import TRAIN_TABLE_NAME
from models.train_classifier import DOCUMENT_TERM_MATRIX_NAME
from models.train_classifier import load_data
from models.train_classifier import load_model
from figures.figures import return_data_figures
from figures.figures import return_result_figures


if len(sys.argv) == 1:
    database_filepath = '../data/disaster_response.db'
    model_filepath = '../models/model.pickle'

elif len(sys.argv) == 3:
    database_filepath, model_filepath = sys.argv[1:]
else:
    print('Please provide the filepaths of the database and model respectively'\
          '\n\nExample: python run.py ../data/disaster_response.db '           \
          '../models/model.pickle')

app = Flask(__name__)

# load training data
_, Y, category_names = load_data(database_filepath, TRAIN_TABLE_NAME)

# load model
model = load_model(model_filepath)

# load document-term matrix
X_term = sp.load_npz('../models/' + DOCUMENT_TERM_MATRIX_NAME)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # return plotly graphs for visualization
    graphs = return_data_figures(X_term, Y, model)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    positive_label_msk = (classification_labels == 1)
    classification_results = dict(zip(
        category_names[positive_label_msk],
        classification_labels[positive_label_msk]
    ))

    # return plotly graphs for visualization
    graphs = return_result_figures(model,
                                   category_names,
                                   classification_labels,
                                   query)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=ids,
        graphJSON=graphJSON
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
