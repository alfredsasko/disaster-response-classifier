'''Script that loads messages and categories data from csv, tranforms them,
and stores them in sqlite database.
'''
# Imports
import sys
import pandas as pd
from sqlalchemy import create_engine


TRAIN_TABLE_NAME = 'train_data'


def load_data(messages_filepath, categories_filepath):
    '''Merge message and categories data from csv files to dataframe'''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, how='inner', on='id')
    return df


def clean_data(df):
    '''Tranform categories column to separate variables'''

    # Split categories column to category label columns
    categories = df['categories'].str.split(';', expand=True)

    # Construct list of category labels
    row = categories.loc[0]
    category_colnames = row.apply(
        lambda category_label: category_label.strip()[:-2]
    )
    categories.columns = category_colnames

    # Make category label columns numeric
    for column in categories:
        categories[column] = (categories[column]
                              .astype(str)
                              .str[-1]
                              .astype(int))

    # Transform categories to binary variables
    non_binary_categories = categories.columns[
        categories.apply(lambda category: len(category.unique())) > 2
    ]

    for category in non_binary_categories:
        # Replace least accurance category codes by most frequent one
        category_codes_distribution = (categories[category]
                                       .value_counts(normalize=True))
        least_frequent_code_list = category_codes_distribution.index[2:]
        most_frequent_code = category_codes_distribution.idxmax()

        categories.loc[categories[category].isin(least_frequent_code_list),
                       category] = most_frequent_code

    # Replace categories column with separate category columns
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()

    return df


def save_data(df, database_filepath, **to_sql_kws):
    '''Saves dataframe to sqlite database'''
    try:
        engine = create_engine('sqlite:///' + database_filepath)
        df.to_sql(con=engine, **to_sql_kws)
    except Exception as exc:
        print('Dataframe not stored:', exc)


def main():
    if len(sys.argv) == 5:

        (messages_filepath,
         categories_filepath,
         database_filepath,
         table_mode) = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving training data in...\n    DATABASE: {}\n    TABLE: {}'
              .format(database_filepath, TRAIN_TABLE_NAME))
        save_data(df, database_filepath,
                  name=TRAIN_TABLE_NAME, if_exists=table_mode, index=False)

        print('Cleaned training data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database and table mode {"fail",'
              '"replace", "append"} as the third and fourth argument.\n\n'
              'Example: python process_data.py disaster_messages.csv '
              'disaster_categories.csv disaster_response.db replace')


if __name__ == '__main__':
    main()
