""" This is the code to get the date & genre data dictionary for the movie project"""

import pandas as pd
import json

# columns to read
usecols = ["id", "imdbId", "Title", "Year of Release", "Genre"]

# read data
data = pd.read_csv('data_plotting.csv', index_col="id", usecols=usecols)


# possible genres
genre_lst = ['Documentary', 'Crime', 'History', 'Drama', 'Family', 'News',
             'Sport', 'Western', 'Fantasy', 'Biography', 'Short', 'Game-Show',
             'Sci-Fi', 'Adventure', 'Animation', 'Musical', 'Comedy',
             'Thriller', 'Music', 'Talk-Show', 'Romance', 'Film-Noir',
             'Reality-TV', 'Horror', 'War', 'Action', 'Adult', 'Mystery']

# possible decades - note: 1980 refers to anything published in 1980 - 1989 inclusive
year_list = []
for year in range(1880, 2030, 10):
    year_list.append(str(year))


# note: there is also an 'All' key in both outer and inner layer which contains all genres/dates
# list of inner and outer layer keys
outer_keys = ['All'] + genre_lst
inner_keys = ['All'] + year_list


# create the empty dictionary - a nested dict of genre and decade of release
genre_to_date_dict = {key: {inner_key: 0 for inner_key in inner_keys} for key in outer_keys}


# helper functions:
def update_by_year(movie_id: int, current_genre: str, counter_dict: dict, movie_data: pd.DataFrame) -> None:
    """Update the counter dictionary's inner layer based on the given genre and movie's year(decade) of release
    note: this also updates the inner All category

    Precondition: counter_dict is the genre_to_date_dict
    """
    release_year = movie_data.loc[movie_id, 'Year of Release']
    # update 'All'          # note this includes even entries without a year of release
    counter_dict[current_genre]['All'] += 1
    # update by year of release             hasattr(release_year, '__len__') and eval(release_year)
    if release_year in range(1880, 2030):
        decade = release_year//10 * 10
        counter_dict[current_genre][str(decade)] += 1


def update_by_genre(movie_id: int, counter_dict: dict, movie_data: pd.DataFrame) -> None:
    """Update the outer layer of the dictionary for each of the relevant genres and also update the outer
    All category.           Precondition: counter_dict is the genre_to_date_dict
    """
    movie_genres = movie_data.loc[movie_id, 'Genre']
    movie_genres = eval(movie_genres)
    # update 'All'
    update_by_year(movie_id, 'All', counter_dict, movie_data)
    # update by genre
    for movie_genre in movie_genres:
        update_by_year(movie_id, movie_genre, counter_dict, movie_data)


# begin counting
# for every movie - from id 0 to 38750 - if it is of [genre], if it is of [decade], update the dictionary
for i in range(0, 38751):
    # update dictionary for that movie
    update_by_genre(i, genre_to_date_dict, data)


# save the resulting dictionary as a universal variable
GENRE_TO_DATE = genre_to_date_dict

# convert to json
with open("genre_to_date.json", "w") as outfile:
    json.dump(genre_to_date_dict, outfile)
