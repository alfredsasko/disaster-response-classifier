│   .gitignore                                # list of ignored files by git
│   README.md                                 # instructions
│
├───app                                       
│   │   run.py                                # flask execution modul
│   │
│   └───templates
│           go.html                           # app landing page
│           master.html                       # app classifier page
│
├───assets                                    # assets for README MD
│
├───data
│       disaster_categories.csv               # raw categories data
│       disaster_messages.csv                 # raw messages data  
│       disaster_response.db                  # app database
│       ETL Pipeline Preparation.ipynb        # design of the ETL pipeline
│       process_data.py                       # ETL script saves app database
│       __init__.py                           # data modul builder
│
├───figures
│       figures.py  	                        # plotly visualizations for app
│       __init__.py                           # figures modul builder
│
└───models
        document_term_matrix.npz  	          # document term sparse matrix
        feature_extraction.py                 # CustomCountVectorizer modul
        ML Pipeline Preparation.ipynb         # desing of the ML pipeline
        model.pickle                          # serialized ML model
        nltk_init.py                          # cusom nltk resources installer
        train_classifier.py                   # ML script saves model and  
                                              # document term matrix
        __init__.py                           # models modul builder
