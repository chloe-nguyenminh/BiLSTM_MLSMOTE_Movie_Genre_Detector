import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/org_data.csv")


def genre_str_to_lst(genre: str):
    return genre.split('|')


df['Genre'] = df['Genre'].apply(genre_str_to_lst)


year_lst = []


def separate_title_year(name: str):
    try:
        year_lst.append(int(name[-6:][1:5]))
        return name[:len(name) - 6]
    except Exception:
        year_lst.append(np.nan)
        return name


df['Title'] = df['Title'].apply(separate_title_year)
# TODO: create new column for year
# TODO: split name from year

df.insert(3, 'Year of Release', pd.Series(year_lst))

df = df.dropna()
df['Year of Release'] = df['Year of Release'].astype(int)

df.to_csv('org_data_after_preprocess.csv')
