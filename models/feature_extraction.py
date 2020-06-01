'''Modul for extraction of custom features from text
    - CustomCoutVectorizer with tagging and lemmatization
    - TextLengthExtractor extracting length of the text
'''

# IMPORTS

# Core libraries
import warnings
from functools import partial
import re

# 3rd party libraries
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from nltk import pos_tag
from nltk.tokenize import sent_tokenize


class CustomCountVectorizer(CountVectorizer):
    '''CountVectorizer with custom word replacment, text tagging
    and token lemmatization. Attribute 'token_counts' holds dictionary
    of token keys and their counts in corpus. Uses method plot to plot
    distribution of tokens.

        Args:
            punct_regex: Regular expressin to match punctuation, None by defaul

            tag_regex_dict: Dictionary with keys as custom tag names and
                values as regular expression used to parse corpus for tagged
                words. Matched strings will be replaced by key (tag)

            replace_regex_dict: Dictionary with keys as custom words
                and values as regular expression used to parse corpus for
                words to be replaced by custom words.

            lemmatizer: callable returning lemmatized token

            **kws: Keyword agruments passed to CountVectorizer init method

            NOTE: tag name need to fullmatch tokenizer pattern otherwise
                warnig is raised. All tags are converted in anlyzer to comply
                  '#' + tag.upper() (ex: cardinaldigit to #CARDNIALDIGIT)

        Attributes:
            token_counts_: Dictionary mapping of tokens to their counts

            _ambiguous_tags: Set of ambiguous tag names in corpus. If not empty
                raises warning to cofound tax name with token having same name.

        Methods:
            plot(self [,topk, sortby, ascending]): plots distribution of tokens
    '''
    def __init__(self, punct_regex=None, tag_regex_dict=dict(),
                 replace_regex_dict=dict(), lemmatizer=None,
                 input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):

        super().__init__(input=input, encoding=encoding,
                         decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor,
                         tokenizer=tokenizer, stop_words=stop_words,
                         token_pattern=token_pattern,
                         ngram_range=ngram_range, analyzer=analyzer,
                         max_df=max_df, min_df=min_df,
                         max_features=max_features, vocabulary=vocabulary,
                         binary=binary, dtype=dtype)

        self.punct_regex = punct_regex
        self.tag_regex_dict = tag_regex_dict
        self.replace_regex_dict = replace_regex_dict
        self.lemmatizer = lemmatizer
        self._ambiguous_tags_ = set()

    def _postprocess(self, doc):
        '''Custom preprocessing of document/text'''

        # Check tag names for ambiguity in doc
        ambiguity_regex = r'|'.join([r'(\b' + tag_name + r'\b)'
                                     for tag_name in self.tag_regex_dict])

        match = re.search(ambiguity_regex, doc, flags=re.IGNORECASE)

        if match:
            self._ambiguous_tags_ = (self._ambiguous_tags_
                                     .union(set(match.groups())))

        # Normalize cases and unicode characters
        preprocess = super().build_preprocessor()
        doc = preprocess(doc)

        # Custom tagging
        if self.tag_regex_dict:
            for tag, regex in self.tag_regex_dict.items():
                doc = re.sub(regex, tag, doc, flags=re.IGNORECASE)

        # Replacement by custom words
        if self.replace_regex_dict:
            for word, replace_regex in self.replace_regex_dict.items():
                doc = re.sub(replace_regex, word, doc, flags=re.IGNORECASE)

        # Remove puncutation
        if self.punct_regex:
            doc = re.sub(self.punct_regex, ' ', doc, flags=re.IGNORECASE)

        return doc

    def build_preprocessor(self):
        '''Build custom preprocessor function'''

        return self._postprocess

    def _check_tags(self, tokenizer):
        '''Checks if tag name matches tokenizer pattern+'''
        for tag in self.tag_regex_dict:
            tokenized_tag = tokenizer(tag)[0]

            assert tag == tokenized_tag, \
                'Can not build tokenizer! Tag {} does not ' \
                'match token pattern. Tokenize tag is {}.' \
                .format(tag, tokenized_tag)

    def _lemmatize(self, doc, tokenizer):
        '''Lemmatize document/text'''

        return [self.lemmatizer(token) for token in tokenizer(doc)]

    def build_tokenizer(self):
        '''Return a custom function that splits a string into
        seuqence of tokens and lemmatize and tag them
        '''
        if self.tokenizer is not None:
            tokenizer = self.tokenizer
        else:
            token_pattern = re.compile(self.token_pattern,
                                       flags=re.IGNORECASE)
            tokenizer = token_pattern.findall

        if callable(self.lemmatizer):
            return partial(self._lemmatize, tokenizer=tokenizer)
        else:
            return tokenizer

    def _pos_tag(self, txt):
        '''return custom part of speach (POS) tag of the text'''

        pos_token = pos_tag([txt])
        token, tag = pos_token[0][0], pos_token[0][1]
        token = self.pos_tag_dict[tag] if tag in self.pos_tag_dict else token

        return token

    def _tag(self, doc, analyzer):
        '''Format tags to according pattern #TAGNAME'''

        return ['#' + token.upper() if token in self.tag_regex_dict else token
                for token in analyzer(doc)]

    def build_analyzer(self):
        '''Return a custom collable that handles preprocessing,
        tokenization, n-grams and tags generation
        '''
        parent_analyzer = super().build_analyzer()

        # Checks if tag names have full match with tokenizer pattern
        self._check_tags(self.build_tokenizer())

        # Handle tagging
        if self.tag_regex_dict:
            return partial(self._tag, analyzer=parent_analyzer)
        else:
            return parent_analyzer

    def fit_transform(self, raw_documents, y=None):
        '''Extend fit tranform method by seting token_counts_ attribute
           and checking ambigious tag names.'''

        X = super().fit_transform(raw_documents, y)

        # Set token_counts_ attribute
        token_names = self.get_feature_names()
        token_counts = np.asarray(X.sum(axis=0)).squeeze()

        token_counts = dict(zip(token_names, token_counts))
        self.token_counts_ = token_counts

        # Check ambiguous tags
        self._ambiguous_tags_ -= {None}
        if self._ambiguous_tags_:
            warnings.warn('Ambiguous tags in corpus found: {}'
                          .format(self._ambiguous_tags_))

        return X

    def plot(self, topk=None, sortby=None, ascending=True, **kws):
        '''Plots bar of token counts.

        Args:
            topk: Integers, plots top k tokens, 20 by default
            sortby: String, whether to sort by 'token' or 'count'.
                None by default means keeping document-term matrix order
            ascending: Boolean, False means sort descanding. True by default.
            title: String, title of the chart None by Default

            **kws: Keyword arguments to pandas.DataFrame.plot function
        '''

        check_is_fitted(self, attributes=['token_counts_'],
                        msg='Token counts are not fitted')

        token_counts = pd.Series(self.token_counts_)

        # Handle sorting
        if sortby == 'token':
            token_counts.sort_index(inplace=True, ascending=ascending)
        elif sortby == 'count':
            token_counts.sort_values(inplace=True, ascending=ascending)

        # Handle title
        if topk:
            token_counts = token_counts[:topk]
            title = 'Distribution of top {} tokens'.format(topk)
        else:
            title = 'Tokens Distribution'

        if 'title' in locals():
            kws['title'] = title if 'title' not in kws else kws['title']

        # Plot bar chart
        token_counts.plot.bar(**kws)

        return None


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''Tranformer extracting text length'''
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_len = X.str.len()
        return X_len.values.reshape(-1, 1)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(self.tokenizer(sentence))
            try:
                first_word, first_tag = pos_tags[0]
            except Exception:
                return False
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
