import pandas as pd
from DataAssignment.src.py.config.config import *
from DataAssignment.src.py.config.querys_recommender import *
import psycopg2
import os.path
import numpy as np
import pandas as pd

# database connection:
try:
    conn = psycopg2.connect(DATABASE_CONN)
except:
    print("Unable to connect to the database")

# database cursor
cur = conn.cursor()


# utility functions
def extract_df_from_query(db_cursor, query, column_types, column_names=None):
    """
    Auxiliary function to parametrize the extraction of the info from the database.

    Parameters
    ----------
    db_cursor : psycopg2.cursor
        cursor connection (opened) with the database
    query: string
        query to extract the information
    column_types: list of string
        column types of the query
    column_names: list of string
        column names of the query

    Returns
    -------
    pandas.DataFrame
    """
    try:
        # runs the query on the database
        db_cursor.execute(query)
        # extracts the info into a list of tuples
        rows = db_cursor.fetchall()
        # convert it to numpy
        np_rows = np.column_stack((
            [[rows[i][j] for i in range(len(rows))] for j in range(len(rows[0]))]
        ))
        # convert it to pandas
        df = pd.DataFrame(np_rows)
        # if column names specified rename them
        if column_names is not None:
            df.columns = column_names
        # na drop
        df = df.dropna()
        # change column types (by default the numpy conversion returns arrays of 'Objects'
        for i, c in enumerate(df.columns):
            df[c] = df[c].astype(column_types[i])

        # eof
        return df
    except psycopg2.ProgrammingError:  # catch any error during query or processing flow
        raise Exception("Unable to execute or process query to the db")


def load_recommender_data(db_cursor, query, path_data):
    """
    Auxiliary function to execute or load recommeder dataset

    Parameters
    ----------
    db_cursor : psycopg2.cursor
        cursor connection (opened) with the database
    query: string
        query to extract the information
    path_data: string
        path to save/load dataset

    Returns
    ---------
    DataFrame
    """
    if os.path.isfile(path_data+'recommender_data.csv'):
        df = pd.read_csv(path_data+'recommender_data.csv', sep=',')
    else:
        df = extract_df_from_query(
            db_cursor, query,
            ['category', 'category', 'int32', 'float32', 'category',
             'category', 'category', 'category', 'category', 'int32', 'float32'],
            ['cust', 'prod', 'target', 'discount', 'source', 'city',
             'country', 'state', 'year_join', 'has_email', 'price'])
        df.to_csv(path_data+'recommender_data.csv', sep=',', index_label=False)
    return df


