import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import csv


with open('movies.csv','r',encoding="utf8") as f:
    movies_Data_Lines = f.readlines()

movies_id = np.zeros([len(movies_Data_Lines), 1], int)
for i in range(1, len(movies_Data_Lines)):
    # movies_id[i-1] = int(( movies_Data_Lines[i] ).split( ',' )[0][0])
    movies_id[i-1] = ((movies_Data_Lines[i]).split(','))[0]

movies_id = -1*movies_id
with open('ratings.csv', 'r') as f:
    user_Data_Lines = f.readlines()
users_data = np.zeros([len(user_Data_Lines), 3])
for i in range(1, len(user_Data_Lines)):
    users_data[i-1] = ((user_Data_Lines[i]).split(','))[0:3]


tmp_float_u_id = np.unique(users_data[:, 0])[1:]
users_id = np.ndarray.astype(tmp_float_u_id,int)
number_of_users = int(len(users_id))
number_of_movies = len(np.unique(movies_id[:, 0]))

number_of_rate_record = np.zeros(number_of_movies,int)
movie_average_rate = np.zeros(number_of_movies,int)
number_of_rated_movie = np.zeros(number_of_users, int)
user_average_rate = np.zeros(number_of_users, int)

for i in range(number_of_movies):
    id = -1*(movies_id[i])
    number_of_rate_record[i] = int(np.count_nonzero(users_data[:, 1] == id))
    temp_inx = np.where(users_data[:, 1] == id)
    if np.any(temp_inx):
        movie_average_rate[i] = np.mean(users_data[temp_inx, 2])

for i in range(number_of_users):
    id = (users_id[i])
    number_of_rated_movie[i] = int(np.count_nonzero(users_data[:, 0] == id))
    temp_inx = np.where(users_data[:, 0] == id)
    if np.any(temp_inx):
        user_average_rate[i] = np.mean(users_data[temp_inx, 2])

G = nx.Graph()
G.add_nodes_from(users_id[:], bipartite=0)
# Add the node attribute "bipartite"
G.add_nodes_from(np.ndarray.tolist(movies_id[:, 0]), bipartite=1)
G.add_edges_from([(users_data[i, 0], -1*users_data[i, 1]) for i in range(len(users_data))])
G.add_edges_from([(-1*users_data[i, 1], users_data[i, 0]) for i in range(len(users_data))])

# nx.clustering(G,node)
cc = nx.square_clustering(G)

# compute page rank
pr = nx.pagerank(G)

# Bipartite Degree Centrality
cen = nx.degree_centrality(G)

# average neighbor degree
avg = nx.average_neighbor_degree(G)

new_data = np.zeros([len(users_data[:, 0]), 15])

for i in range(len(users_data[:, 0])-1):
    d = users_data[i, :]
    movie_index = - d[1]
    user_index = d[0]
    tt = np.where(users_id == user_index)
    mm = np.where(movies_id == movie_index)[0]
    a = number_of_rate_record[mm]
    new_data[i, :] = [d[0], pr[user_index], cc[user_index], cen[user_index], avg[user_index], user_average_rate[tt], movie_average_rate[mm], number_of_rated_movie[tt], -d[1], pr[movie_index], cc[movie_index], cen[movie_index], avg[user_index], number_of_rate_record[mm], d[2]]

with open('my_data.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(new_data)


