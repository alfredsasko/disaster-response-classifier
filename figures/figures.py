'''This modul contains support to wrangle data and build figures of the app'''

import sys
import time
import json

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.utils.validation import check_is_fitted

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from wordcloud import WordCloud

# Wordcloud parameters
MAX_WORDS = 100
BACKGROUND_COLOR = 'white'
LABELS = False    # lables=True, add dropdown manu in wordcloud per category

# Model explanability parameters
TOPK_COEF = 10     # Number of top k weights sorted by their size

# Drop down manu parameters
ACTIVE_ITEM = 0

def return_data_figures(X, Y, model):
    '''Creates figures for plotly visualization related to training dataset

    Args:
        X: Sparse matrix of document-term matrix of shape (# documents, # terms)
        Y: Dataframe of labels as binary matrix of shape (# documents, # labels)
        model: Fitted GridSearchCV object

    Returns:
        figures: List of dictionary representation of plotly figure objects
    '''

    check_is_fitted(model)

    # bar chart and word cloud : Distribution of Document Categories
    category_names = Y.sum().index
    category_counts = Y.sum().sort_values(ascending=False)


    fig_category = go.Figure(
            data=go.Bar(
                x=category_names,
                y=category_counts,
                marker_color=DEFAULT_PLOTLY_COLORS[0],),
            layout_title_text='Distribution of the Message Categories',
            layout_title_x=0.5,
            layout_yaxis_title='Count'
    )

    # Word Cloud: Distribution of MAX_WORDS Most Frequent Words in Corpus

    title_text = 'Top {} Most Frequent Words in the Corpus'.format(MAX_WORDS)

    fig_term = go.Figure(
        layout_title_text=title_text,
        layout_title_x=0.5,
        layout_xaxis_showticklabels=False,
        layout_yaxis_showticklabels=False,
        # layout_paper_bgcolor='LightSteelBlue',
        layout_margin=dict(b=50)
    )

    # Generate dictionary of WordCloud objects per label/category
    buttons = []

    term_cloud_dict = get_term_cloud(
        X, Y, model,
        labels=LABELS,
        max_words = MAX_WORDS,    # change to get topk words in the wordcloud
        background_color=BACKGROUND_COLOR)   # change wordcloud background color

    # Add dropdown to get word clouds per selected category
    dropdown_item_ar = np.array(list(term_cloud_dict.keys()))

    for category, term_cloud in term_cloud_dict.items():

        if category=='all':
            title = title_text
        else:
            title = (
                'Top {} Words for "{}" Category'
                .format(MAX_WORDS, category)
            )

        fig_term.add_trace(
            go.Image(z=term_cloud,
                     name=category,
                     visible=True if category == 'all' else False)
        )

        buttons.append(
            dict(method='update',
                 label=category,
                 args=[{'visible': dropdown_item_ar == category},
                       {'title': title}]
        ))

    fig_term.update_layout(
        updatemenus=[
            dict(
                visible = True if LABELS else False,
                active=ACTIVE_ITEM,
                buttons=list(buttons),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.2,
                yanchor="top"
            )
        ]
    )

    # Stack figures
    figures = [fig_category, fig_term]

    return figures


def get_term_cloud(X, Y,  model, labels=False, **kws):
    '''Returns dictionary of wordcloud objects per each label

    Args:
        model: Fitted GridSearchCV object
        X: Sparse matrix of document-term matrix of shape (# documents, # words)
        Y: DataFrame of labels as binary matrix of shape (# documents, # labels)
        labels: If True return returnbs word cloud per each label.
            False by default return only wordcloud for all labels
        kws: Keyword arguments to WorldCloud constructor

    Returns:
        wordcloud_dict: Dictionary with key value pairs as {label: WorldCloud}
            of size (# of labels)
    '''

    # Get words in the corpus
    term_names = (model
                  .best_estimator_.named_steps['vectorizer']
                  .get_feature_names())

    # Filter out tags
    tag_msk = pd.Index(term_names).str.contains('#')

    wordcloud_dict = {}

    # Insert world cloud for whole corpus
    all_category_term_counts = pd.Series(
        np.asarray(X.sum(axis=0)).squeeze(),
        index=term_names
    )

    wordcloud_dict['all'] = (
        WordCloud(**kws)
        .generate_from_frequencies(all_category_term_counts[~tag_msk].to_dict())
    )

    if labels:

        # Fitler out zero count categories
        zero_count_category_msk = (Y.sum() == 0)
        category_list = Y.columns[~zero_count_category_msk]

        for category in category_list:

            # Filter documents by category
            document_idx = Y[category][Y[category] == 1].index

            # Calculate term frequencies per category
            term_counts = (
                np.asarray(
                    sp.lil_matrix(X)
                    [document_idx, ]
                    .sum(axis=0))
                .squeeze()
            )

            term_counts = pd.Series(term_counts, index=term_names)

            # Create and fit worldcloud object
            wordcloud_dict[category] = (
                WordCloud(**kws)
                .generate_from_frequencies(term_counts[~tag_msk].to_dict())
            )


    return wordcloud_dict


def return_result_figures(model, category_names, classification_labels,
                          document):
    '''Creates figures for plotly visualization to explain classifier decision

    Args:
        model: Fitted GridSearchCV object
        category_names: List of label/classes names of size (# of labels/classes)
        classification_labels: binary array of multi-label classification
            of size (# of labels/classes)
        document: String as document/message
        Note: 0-negative 1-positive label/class prediction

    Returns:
        figures: List of dictionary representation of plotly figure objects
    '''

    check_is_fitted(model)

    # Get words in the corpus
    term_ar = np.array(model
                       .best_estimator_.named_steps['vectorizer']
                       .get_feature_names())

    # Generate dictionary of WordCloud objects per label/category
    buttons = []

    # Add dropdown to get explation per label
    positive_label_msk = (classification_labels == 1)
    estimator_ar = (
        np.array(
            model
            .best_estimator_
            .named_steps['classifier']
            .estimators_
        )
        [positive_label_msk]
    )

    label_ar = category_names[positive_label_msk]

    # Design layout
    fig_imp = make_subplots(
        rows=1, cols=2,
        subplot_titles=('TOP {} Detractors / Supporters'.format(TOPK_COEF),
                        'Message Detractors / Supporters')
    )

    fig_imp.update_layout(
            title_text = ('Explanation of Classification Result for '
                          + 'the Category "{}"'.format(label_ar[ACTIVE_ITEM])),
            title_x=0.5
    )
    fig_imp.update_xaxes(title_text='Importance: - detractors / + supporters')
    fig_imp.update_yaxes(title_text='Words', col=1, row=1)

    for i, (label, estimator) in enumerate(zip(label_ar, estimator_ar)):

        # get TOPK coeficients for estimator
        coef_ar = estimator.named_steps['svm'].coef_[0]
        selected_term_msk = estimator.named_steps['selector'].get_support()

        coef_sr = pd.Series(coef_ar, index=term_ar[selected_term_msk])
        sorted_coef_idx = coef_sr.abs().sort_values(ascending=False).index
        imp_sr = coef_sr[sorted_coef_idx][:TOPK_COEF].sort_values()

        # Add trace for TOPK coeficients per label
        fig_imp.add_trace(
            go.Bar(x=imp_sr,
                   y=imp_sr.index,
                   orientation='h',
                   name=label,
                   marker_color=DEFAULT_PLOTLY_COLORS[0],
                   visible=True if i == ACTIVE_ITEM else False),
            row=1, col=1
        )

        # get coeficients of document/message terms/tokens
        term_vec = (model
                    .best_estimator_
                    .named_steps['vectorizer']
                    .transform([document])
                    .toarray()
                    .squeeze())

        term_vec_msk = (term_vec == 1)
        print(term_ar[term_vec_msk])

        term_sr = coef_sr[coef_sr.index.isin(term_ar[term_vec_msk])]
        print(term_sr, file=sys.stderr)
        sorted_term_idx = term_sr.abs().sort_values(ascending=False).index
        term_sr = term_sr[sorted_term_idx].sort_values()

        fig_imp.add_trace(
            go.Bar(x=term_sr,
                   y=term_sr.index,
                   orientation='h',
                   name=label,
                   marker_color=DEFAULT_PLOTLY_COLORS[0],
                   visible=True if i == ACTIVE_ITEM else False),
            row=1, col=2
        )

        buttons.append(
            dict(method='update',
                 label=label,
                 args=[{'visible': label == np.repeat(label_ar, 2)},
                       {'title': ('Explanation of Classification Result '
                                  + 'for the Category "{}"').format(label)}]
        ))

    fig_imp.update_layout(
        updatemenus=[
            dict(
                active=ACTIVE_ITEM,
                buttons=list(buttons),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.3,
                yanchor="top"
            )
        ],
        showlegend=False
    )

    # Stack figures
    figures = [fig_imp]

    return figures
