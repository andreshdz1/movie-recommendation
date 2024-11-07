import pandas as pd
import pickle

# movie of the user
movie_name = "The Matrix"

# list of movies
full_movies = pd.read_csv("../data/imdb_top_1000.csv")
numeric_movies = pd.read_csv("../data/numeric_movies.csv")

reference_movie = full_movies[full_movies["Series_Title"] == movie_name]
index_movie = reference_movie.index.item()
numeric_movie = numeric_movies.iloc[index_movie]

# load model
with open('../notebooks/model.pkl', 'rb') as f:
    model = pickle.load(f)

recommendation = model.kneighbors(numeric_movie.values.reshape(1, -1))
recommended_movies = full_movies.iloc[recommendation[1][0][1].item()]["Series_Title"]
print("Your set of movies: ", recommended_movies)