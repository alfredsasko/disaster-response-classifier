│   .gitignore                                # list of the ignored files by git
│   README.md                                 # instructions
│
├───app                                       
│   │   run.py                                # flask execution module
│   │
│   └───templates
│           go.html                           # application landing page
│           master.html                       # application classifier page
│
├───assets                                    # assets for the README file
│
├───data
│       disaster_categories.csv               # raw categories data
│       disaster_messages.csv                 # raw messages data  
│       disaster_response.db                  # application database
│       ETL Pipeline Preparation.ipynb        # design of the ETL pipeline
│       process_data.py                       # ETL script saves app database
│       __init__.py                           # data module builder
│
├───figures
│       figures.py  	                        # plotly visualizations for app
│       __init__.py                           # figures module builder
│
└───models
        document_term_matrix.npz  	          # document-term sparse matrix
        feature_extraction.py                 # CustomCountVectorizer module
        ML Pipeline Preparation.ipynb         # design of the ML pipeline
        model.pickle                          # serialized ML model
        nltk_init.py                          # custom nltk resources installer
        train_classifier.py                   # ML script saves model and  
                                              # document-term matrix
        __init__.py                           # models module builder
