'''Modul trains disaster response classifier'''

# IMPORTS

# Core libraries
import sys
import warnings
import re
import pickle
import ipdb

# Numerical computation and statistics
import numpy as np
import pandas as pd

# Databases
from sqlalchemy import create_engine

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# Feature selection
from sklearn.feature_selection import SelectFromModel

# Estimators
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

# Pipelines
from sklearn.pipeline import Pipeline

# Multi-label classification
from sklearn.multiclass import OneVsRestClassifier

# Evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Custom libraries
from feature_extraction import CustomCountVectorizer

# Natural language processing
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk_init import InitNLTK
nltk_resources = {
    'punkt': 'tokenizers/punkt',
    'wordnet': 'corpora/wordnet',
    'stopwords': 'corpora/stopwords',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    'omw': 'corpora/omw'
}
InitNLTK(nltk_resources).download_resources()

def load_data(database_filepath, table_name):
    '''Loads table from database and returns input
    and output dataframe.

    Args
        database_filepath: String, location of database
        table_name: String, name of table to load

    Returns:
        X: Dataframe, input variables, of shape (# samples, # features)
        Y: Dataframe, output variables, of shape (# samples, # targets )
        category_names: Index, with category/target names
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, con=engine)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names

def build_vectorizer():
    """
    Build CustomCountVectorizer instance (replacment of tokenizer function
    in project). It enables custom tagging, token replacment, tokenizing
    and lemmatizing.

    Returns:
    vectorizer: Initiated CustomCountVectorizer
    """
    # Regext for removing  punctuation
    punct_regex = r'[^a-zA-Z]'

    # Custom tags
    tag_list = ['urladdress',
                'yeardate',
                'kilounit',
                'cardinaldigit']

    # Regex expression matching custom tags
    regex_list = [
        r'(?:(?:(?:URL:?)?http[s]?\s?:?\s?(?://)?(?:www.)?)|(?:www))' \
        r'(?:(?:(?:bit.ly|ow.ly|j.mp|tinyurl.com|tr.im|tl.gd'         \
        r'|goo.gl\s?fb|ur14.eu|su.pr|ff.im|goto.gg|uurl.in|url.ie'    \
        r'|digg.com|rep.ly|twitpic.com|nxy.in)\s?)|[a-zA-Z]|[0-9]'    \
        r'|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'(?:19[0-9]{2})|(?:20(?:(?:[0-1][0-9])|(?:20)))',
        r'[1-9](?:[0-9])*\s?,?\s?000',
        r'(?:\b[1-9][0-9]*(?:,[0-9]+)?(?:\.\s?[0-9]+)?)|(?:\btwo\b)'
    ]

    # Custom tagging of url addresses, yeardate, kilonunits and cardinaldigits
    # reduces dimensionality of document-term matrix by ~16%
    tag_regex_dict = dict(zip(tag_list, regex_list))

    # Regex expressions for replacing slang words (replace 'u' with 'you')
    slang_regex_dict = {'you': r'(?<!@)\bu\b(?!(?:\.\s?\w\.)+)(?!\s\w\s)(?!@)'}

    # Update stopwords by CountVectorizer warning suggestion
    stop_words = stopwords.words('english')
    stop_words.extend(["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha',
                      'might', 'must', "n't", 'need', 'sha', 'wa', 'wo',
                      'would', 'n'])

    # Instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # CustomCountVectorizer uses custom tagging and default tokenizer
    # with tokens matching r'(?u)\b\w\w+\b', it is much faster then
    # nltk.word_tokenize function.
    vectorizer = CustomCountVectorizer(
        punct_regex=punct_regex,
        tag_regex_dict=tag_regex_dict,
        replace_regex_dict=slang_regex_dict,
        lemmatizer=lemmatizer.lemmatize,
        tokenizer=None,
        stop_words=stop_words,
        max_features=None
    )

    return vectorizer


def build_model():
    '''Build pipeline and design model tuning.

    Returns:
        cv: GridSearchCV object
    '''

    random_state = 1

    # Build pipeline
    pipeline = Pipeline([

        # Create document-term matrix
        ('vectorizer', build_vectorizer()),

        # Term Frequency Inverse Document Frequency Transformation
        ('tfidf', TfidfTransformer()),

        # Multi-label model with feature selection
        ('classifier', OneVsRestClassifier(
            estimator=Pipeline([

                # Feature selection based on L1 regularization
                ('selector', SelectFromModel(
                    LinearSVC(penalty='l1',
                              dual=False,
                              tol=1e-3,
                              random_state=random_state)
                )),

                # Support Vector Machine model with weight balancing to
                # remedy unbalanced dataset
                ('svm', LinearSVC(class_weight='balanced',
                                  dual=False,
                                  max_iter=1000,
                                  random_state=random_state))
            ]),
            n_jobs=-1
        ))
    ])

    # Using bi-grams significantly imporved f1_macro score but introduced
    # severe overfitting. Using feature selection, tuning of penalty parameter C
    # have succeeded in reducing model overfitting from 50% to 17% while model
    # f1_macro score reduced slightly from 46% to 42%.
    parameters= {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__estimator__svm__C': [0.01, 0.1, 1]
    }

    # f1_macro score was selected as center of the interest is ability of the
    # model to distinct between labels, while considering labels equal weight
    cv = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring='f1_macro',
        cv=3,
        return_train_score=True,
        verbose=3,
        n_jobs=1)    # change to -1 if there is enough memory and disk-space

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Show precision, recall, and f1-score of the model'''

    Y_true = Y_test
    Y_pred = model.predict(X_test)

    print(classification_report(Y_true, Y_pred, target_names=category_names))

    return None


def save_model(model, model_filepath):
    '''Saves the model as a pickle file'''

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 4:
        database_filepath, table_name, model_filepath  = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath, table_name)

        const_feature_msk = Y.columns[Y.columns.str.contains('child_alone')]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model on train sample...')
        evaluate_model(model, X_train, Y_train, category_names)

        print('Evaluating model on test sample...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, table name as second argument and the ' \
              'filepath of the pickle file to save the model to as the '      \
              ' third argument. \n\nExample: python train_classifier.py ' \
              '../data/disaster_response.db train_data model.pickle')

if __name__ == '__main__':
    main()
