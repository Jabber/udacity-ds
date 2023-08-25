import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads data from csv files and merges them into a single dataframe as well as two separate dataframes for
    further cleaning opportunities

    :param messages_filepath: filepath to messages csv file
    :param categories_filepath: filepath to categories csv file

    :return:
    df          unified dataframe,
    categories  dataframe for cleaning,
    messages    dataframe for cleaning
    '''
    # read 2 datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories)
    return df, categories, messages


def clean_data(df, categories, messages=[]):
    '''
    Cleans data by expanding categories and assigning 0-1 values, removing unused columns and duplicates

    :param df: unified dataframe to work with
    :param categories: dataframe to clean up values
    :param messages: optional param in case other modulations are required for messages sub-dataframe

    :return:
    df          unified and clean disaster messages + categories dataframe
    '''
    # expand categories column to multiple columns
    test_text = str(categories.categories[0]).split(';')
    category_colnames = [category.split('-')[0] for category in test_text]
    categories_id = categories.id
    categories = categories.categories.str.split(';', expand=True)
    categories.columns = category_colnames

    # clean values to become 0/1 numbers
    for column in categories:
        categories[column] = categories[column].apply(lambda x: str(x)[-1:])
        categories[column] = pd.to_numeric(categories[column], downcast='integer')
    categories['id'] = categories_id.copy()

    # remove outlier values based on jupyter notebook
    categories['related'] = categories['related'].replace(to_replace=2, value=1)

    # clean up table
    df = df.drop('categories', axis=1)
    df = df.merge(categories, how='outer', on=['id'])
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Saves data into SQL database for future retrieval

    :param df: unified disaster messages + categories dataframe
    :param database_filename: target file to write DB in
    :return: None
    '''
    # save into sql db
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories, messages = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories, messages)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
