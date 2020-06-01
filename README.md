# Disaster Response Workflow Tool

Do you want to integrate machine learning into your organization and make sure that it maximizes impact on employee morale, customers, and your bottom line? This repository is supplemental material to medium series [How to effectively adopt machine learning in organization?](#)

### Table of Contents
1. [Project Motivation](#motivation)
2. [Results](#results)
4. [Installation](#installation)
3. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Project Motivation<a name="motivation"></a>
Machine learning has become very popular in recent years. The entrance costs to the field are low with the rising popularity of massive open online courses. Most of them focus on model building, very few on model deployment. To my knowledge no of them cover the complete ML integration lifecycle in the organization beyond CRISP-DM.

The purpose of the series is to provide a practical recipe on how to make this amazing technology adoption in an organization successful step by step.

The purpose of this repository is to share with fellow data scientists on how to design and deploy Natural Language Processing application following the CRISP-DM process.

## 2. Results<a name="results"></a>
Passing all CRISP-DM phases you should end up with deployed _Disaster Response Workflow Tool_ using Elastic Beanstalk (AWS). You can have fun and play with a tool [here](http://drp-app-prod.eu-central-1.elasticbeanstalk.com/). Refer to this [series article](#) for more details.

### 2.1 Business Understanding
During natural disasters, response teams are overwhelmed by thousands of messages either directly or through social media. They need to filter relevant requests, analyze, prioritize them to make sure that proper organization responds timely to help impacted individuals.

Empowering response teams with the workflow tool based on NLP technology would free up the resources and enable teams to react faster and so saving more lives and reducing financial loss from potential damages of public and private properties.

### 2.2 Data Understanding
The dataset provided by [Figure Eight](https://appen.com/datasets/combined-disaster-response-data/) contains 30000 messages and news articles drawn from 100s of different disasters. The messages have been classified into 36 different categories related to disaster response.

### 2.3 Data Preparation
[Messages](https://github.com/alfredsasko/disaster-response-workflow-tool/blob/master/data/disaster_messages.csv) and [categories](https://github.com/alfredsasko/disaster-response-workflow-tool/blob/master/data/disaster_categories.csv) are stored in separate csv files. They are feeded to [ETL pipeline](https://github.com/alfredsasko/disaster-response-workflow-tool/blob/master/data/process_data.py) which cleans, merge and save data to [sqlite database](https://github.com/alfredsasko/disaster-response-workflow-tool/blob/master/data/disaster_response.db). To run the ETL pipeline clone the repository and run script in terminal providing csv files paths, database filepath and table mode respectively:

```
cd path/to/cloned/repository/data
python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db replace
```

### 2.4 Modeling
The business challenge can be translated into a classic document classification task, where each document (message) can be labeled by one or more labels out of 36 categories. It is a Multi-Label classification problem which might be approached from [three perspectives](https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/):

__1. Multiple-label Problem Transformation__ to the single-label task using _Binary relevance_, _Classifier Chains_ or _Label Powerset_

__2. Adapted Algorithms__ to directly perform multi-label classification rather than transforming the problem into single-label

__3. Ensemble Approaches__ that construct an ensemble of base multi-label classifiers.

Referring to deployment requirements prioritizing speed, memory usage, and fast retraining of the model as new messages feed in over model performance the binary relevance method has been chosen.

Classifier chains would be beneficial as some labels are highly correlated but it takes a long time to train.

Label Powerset is not practical due to a high number of categories. It would highly increase the unbalance ratio of the dataset, where some of the transformed classes would have only one observation.

Even if adapted algorithms and ensemble approaches would most probably improved model performance they are not an option due to high memory usage and long training time.

#### 2.4.1 Performance Metrics
As it is equally important to correctly identify the true danger to human life (recall) as well as the false danger (precision) to safe limited resources which are usually missing during the disaster, the __f1 score__ matric was selected to measure model performance.

The objective is to improve model discriminant power among 36 categories. The same weight needs to be put on each category during learning ignoring the highly skewed distribution of categories in the dataset. This can be done using `f1_macro` score and selecting ML algorithms more robust to unbalance datasets.

The __reference model__ is `DummyClassifier` generating predictions uniformly at random `strategy=uniform` and putting the same weight on each class.

#### 2.4.2 Model Screening
Two simple classifiers comply with the requirements of the lightweight web app and can effectively deal with unbalanced datasets. `LinearSVC` with `class_weight='balanced'` and `ComplementNB`.

| Model        	| mean_train_f1_macro | mean_test_f1_macro |
|--------------	|:-------------------:|:------------------:|
| LinearSVC    	|         0.898       |        0.434       |
| ComplementNB 	|         0.420       |        0.292       |
| Reference    	|         0.122       |        0.120       |

_Linear Support Vector Classifier_ performance is best compared to Naive Bayes and reference. It was selected for further hyper-parameters tunning to improve performance and reduce severe overfitting.

#### 2.4.3 Model Tunning

##### Improve performance
Quantitative parameters as `text_length`, `genre`, `starting_verb`, and `ngram_range` have been tested to improve performance in a grid search. Using bi-grams has proven to be beneficial for performance, but slightly increased overfitting.
The quantitative parameters were not important at all.

Note: orange = train, blue = test
<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/01_model_tunning_ngrams_genre.png" style="float: left; width: 100%;"/>
</div>
<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/02_model_tunning_startverb_genre.png" style="float: left; width: 100%;"/>
</div>
<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/03_model_tunning_textlength_genre.png" style="float: left; width: 100%;"/>
</div>
<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/04_model_tunning_startverb_ngrams.png" style="float: left; width: 100%;"/>
</div>
<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/05_model_tunning_textlength_ngrams.png" style="float: left; width: 100%;"/>
</div>
<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/06_model_tunning_textlength_ngrams.png" style="float: left; width: 100%;"/>
</div>

##### Overfitting Reduction
Feature selection based on _L1 regularization_ has been selected to reduce the number of features from ~26000 to ~4000. _Penalty parameter C_ was tuned as well with `ngram_range`.

There is an interaction with feature selection and ngram_range. With no feature selection (ref. Improve performance) using bi-grams improved performance, but with feature selection using bi-grams deteriorate performance.

Reducing the penalty parameter has a huge impact on reducing overfitting.

For the current dataset best model is using feature selection with L1 regularization, uni-grams and penalty parameter `C=0.1`

<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/07_model_tunning_ngrams_c.png" style="float: left; width: 100%;"/>
</div>

#### 2.4.4 Model Retraining
If the training dataset is updated it is possible to retrain the model using ML pipeline which uses a grid search with the following parameters `ngram_range = [(1, 1), (1, 2)]` and `C=[0.01, 0.1, 1]`.

You need to provide a path to the database and path and name of the model file. Script serializes fitted `GridSearchCV` object to pickle file:

```
cd path/to/cloned/repository/models
train_classifier.py ../data/disaster_response.db model.pickle

```

### 2.5 Deployment
#### 2.5.1 requirements
Thousands of messages need to be analyzed each day during a disaster. The service needs to be quite fast and scalable. Therefore it is favorable to use simple ML models that do not use a lot of memory to have fast prediction time and training is not computationally expensive.

On the other hand the performance of those models can be worse which can be improved by using a more frequent re-training cycle. This would recure functionality to enable users to correct wrong classification results to further improve model performance.

#### 2.5.2 Demo
To engage users to see the model in the action flask web application was developed and deployed using Elastic Beanstalk (AWS). You can try it [here](http://drp-app-prod.eu-central-1.elasticbeanstalk.com/). It is possible to submit a message to classify and shows statistics of training dataset as label distribution and word frequencies as a word cloud.

<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/08_workflow_tool_landing.png" style="float: left; width: 100%;"/>
</div>

The entered message is labeled by 1 or more labels out of 36 categories. App also explains the reasons for each predicted label. It shows the top 10 words as detractors and supporters for each label that the user can see how the model recognizes each class. The app parses the message and explains how each word contributes to a particular label selection.

<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/09_workflow_tool_classify.png" style="float: left; width: 100%;"/>
</div>

#### 2.5.3 Production
To integrate model to production-ready workflow tool the three RESTful APIs in Flask should be developed.
  1. `predict` function to return JSON including 36 categories and their binary indicators
  2. `explain_label` function to return JSON with top k words and their model coefficients.
  3. `explain_message` function to parse entered message and return message words and their model coefficients.

This is out of the scope of the project.

## 3. Installation <a name="installation"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores in the database
        ```
        cd data
        python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db replace
        ```
    - To run ML pipeline that trains classifier and saves
        ```
        cd ../models
        python train_classifier.py disaster_response.db model.pickle
        ```

2. Run the following command in the app's directory to run your web app.
    ```
    cd ../app
    python run.py
    ```

3. Go to http://0.0.0.0:3001/

## 4. File Descriptions <a name="files"></a>
<div style="overflow: hidden; padding: 20px 0px">
    <img src="/assets/10_project_tree.png" style="float: left; width: 100%;"/>
</div>

## 5. Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to [Figure Eight Technologies](https://appen.com/) for the data,  @udacity for the starter code, and Shubham Jain @shubhamjn1 for nice [introduction to multi-label classification](https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/).
