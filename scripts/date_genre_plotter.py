"""Plot the data calculated in date_genre_count_data regarding the relation of movie genre and dates.

Contains the function plot_movies_of_genre_over_decades() which allows you to plot the decade of release vs number of
movies for a given genre.

You can choose a graph type of plot and bar, can choose a custom title, and choose whether to include all instances of
movies for a genre (regardless of release date) in a graph besides the other categories.
To overlap more than one graph, run the function several times for different genres and make sure to set show= False
except for the last one. Alternatively, you can set show=False for everything and put plt.show() in the end.
"""

from date_genre_count_data import GENRE_TO_DATE as DATA
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def test_func1():
    """test function to see if the imported dictionary works --> it does """
    print(DATA['Mystery']['All'])


def plot_movies_of_genre_over_decades(genre: str, movie_data: dict, graph_type: str,
                                      include_all_dates_as_instance: bool = False, show: bool = True,
                                      custom_title: Optional[str] = None) -> None:
    """Plots a graph of the number of movies of the given genre to date based on the given date and genre.
    Note: we igonore movie_data[genre]['All'] unless include_all_dates_as_instance == True
    Note: if show is False, doesn't run plt.show() in the end. This is useful for making several graphs at once

    Preconditions:
    - movie_data is the same as GENRE_TO_DATE or DATA
    - graph_type in ['bar', 'plot']
    - genre is a valid genre or 'All'
    """
    # xpoints setup
    year_list = []
    for year in range(1880, 2030, 10):
        year_list.append(str(year))

    if include_all_dates_as_instance:
        year_list = ['All'] + year_list

    # ypoints setup
    num_movies = []
    for year in year_list:
        num_movies += [movie_data[genre][year]]

    # define xpoints and ypoints
    xpoints = np.array(year_list)       # decade
    ypoints = np.array(num_movies)      # number of movies

    # title
    if custom_title is not None:
        plt.title(custom_title)

    # labels
    plt.xlabel("Decade of Release")
    plt.ylabel("Number of Movies")

    # plot graph / chart
    if graph_type == 'plot':
        plt.plot(xpoints, ypoints, label=genre)
    elif graph_type == 'bar':
        plt.bar(year_list, num_movies, label=genre)
    else:   # assume it was a graph
        plt.plot(xpoints, ypoints)

    # legend
    plt.legend(loc="upper right")

    if show:
        plt.show()


# Some Tests / Demos

def test_plot_movies1():
    """function to test plot_movies_of_genre_over_decades.
    Plot the changes in movies over time in genres of Action, Mystery, Animation and All in that order"""
    plot_movies_of_genre_over_decades('Action', DATA, 'plot', True, show=False)
    plot_movies_of_genre_over_decades('Mystery', DATA, 'plot', True, show=False)
    plot_movies_of_genre_over_decades('Animation', DATA, 'plot', True, show=False)
    plt.show()
    plot_movies_of_genre_over_decades('All', DATA, 'plot', True, True)


def test_plot_movies2():
    """function to test plot_movies_of_genre_over_decades.
    Plot the changes in movies over time in genres of Action and Horror as bar charts"""
    # TAKEAWAY: More than 1 barcharts don't work as well compared to plots
    # Takeaway2 --> NOW FIXED: the legend shows up empty for bar charts
    # Takeaway3: it sticks to the last custom_title defined before plt.show()
    plot_movies_of_genre_over_decades('Action', DATA, 'bar', True, show=False, custom_title="Hello World!")
    plot_movies_of_genre_over_decades('Horror', DATA, 'bar', True, show=True, custom_title="Action & Horror Over Time")


def test_plot_movies3():
    """function to test plot_movies_of_genre_over_decades.
    Test to see what happens if you try to overlap a barchart and graph.
    Also that include_all_dates_as_instance is True for one and False for the other.
    Please don't do this. """
    plot_movies_of_genre_over_decades('Action', DATA, 'plot', True, show=False)
    plot_movies_of_genre_over_decades('Horror', DATA, 'bar', False, show=True)
