import sys
import pandas as pd
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Loads data from provided DB file into input variable and output categories
    :param database_filepath: filepath for SQL database to open
    :return:
    X   dataframe with messages to use as input variable
    y   dataframe with predictable output values
    '''
    # read sql db
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)

    # split into input and output variables, Y being the 36 categories
    X = df.message
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, y


def tokenize(text):
    '''
    Process text with tokenization and lemmatization and standardization
    :param text: string input to tokenize
    :return:
    clean_tokens    a list of tokens that can be used for modelling
    '''
    # tokenize to words
    tokens = word_tokenize(text)

    # lemmatize for words, verbs
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    clean_tokens = [lemmatizer.lemmatize(tok, pos='v').lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    '''
    Create ML pipeline for modelling connection between input and output variables
    :return:
    pipeline    scikit Pipeline object to fit and predict
    '''
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, y_test):
    '''
    Use model to predict outcomes and evaluate accuracy
    :param model: scikit Pipeline object to use for prediction
    :param X_test: input variable test split
    :param y_test: output variables test split
    :return:
    Prints out classification reports
    '''
    # predict categories
    y_pred = model.predict(X_test)

    # print classification reports
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    '''
    Saves model into a pickle file
    :param model: fitted model to reuse in the future
    :param model_filepath: target filepath to save the model to
    :return: None
    '''
    # save to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
