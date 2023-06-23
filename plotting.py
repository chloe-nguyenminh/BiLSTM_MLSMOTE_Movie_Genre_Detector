import pandas as pd
from string import whitespace
import matplotlib as plt
df = pd.read_csv('data_plotting.csv')
#
# genre_set = set()
# for genre in df['Genre']:
#     genre = genre.strip('][').split(',')
#     for g in genre:
#         g = g.strip('"')
#         g = g.strip("'")
#         g = g.strip(whitespace + '"\'')
#         genre_set.add(g)
#         a = 1

genre_lst = ['Documentary', 'Crime', 'History', 'Drama', 'Family', 'News',
             'Sport', 'Western', 'Fantasy', 'Biography', 'Short', 'Game-Show',
             'Sci-Fi', 'Adventure', 'Animation', 'Musical', 'Comedy',
             'Thriller', 'Music', 'Talk-Show', 'Romance', 'Film-Noir',
             'Reality-TV', 'Horror', 'War', 'Action', 'Adult', 'Mystery']
print(genre_lst)
x_axis = [i for i in range(1900, 2020, 10)]
grand_dict = {'time': x_axis}
for genre in genre_lst:
    y_axis = []
    for i in x_axis:
        for movie in df:
            pass
    grand_dict[genre] = y_axis





