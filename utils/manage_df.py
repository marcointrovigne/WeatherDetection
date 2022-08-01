import pandas as pd

"""
General methods to manipulate df
"""


def create_split_df(path_txt):
    """
    Create Dataframe from .json file
    :param path_txt: path to txt file in which there are the names for training, validation or testing
    :return: df with all the filenames in .json format
    """

    frame = pd.DataFrame(columns={"filename"})
    with open(path_txt) as f:
        lines = f.readlines()
        lines = [i.replace('\n', '.json') for i in lines]
        lines = [i.replace(',', '_') for i in lines]
        for line in lines:
            df1 = pd.DataFrame.from_dict({"filename": [line]})
            frame = frame.append(df1, ignore_index=True)
    return frame


def df_to_alias(df, dict_df):
    """
    Convert string into corresponding index in the dict_df
    :param df: df that you want to convert
    :param dict_df: dict_df
    :return: df with int values
    """

    df['daytime'] = df['daytime'].map(lambda a: dict_df['daytime_alias'][a])
    df['precipitation'] = df['precipitation'].map(lambda b: dict_df['precipitation_alias'][b])
    df['fog'] = df['fog'].map(lambda c: dict_df['fog_alias'][c])
    df['roadState'] = df['roadState'].map(lambda d: dict_df['roadState_alias'][d])
    df['sidewalkState'] = df['sidewalkState'].map(lambda e: dict_df['sidewalkState_alias'][e])
    df['infrastructure'] = df['infrastructure'].map(lambda f: dict_df['infrastructure_alias'][f])
    return df
