# Import necessary libraries
import numpy as np
import pandas as pd

# Explore Kaggle input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load movie and credits data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Display the first 2 rows of the movies dataframe
movies.head(2)

# Display the shape of the movies dataframe
movies.shape

# Display the first few rows of the credits dataframe
credits.head()

# Merge movies and credits dataframes on 'title'
movies = movies.merge(credits, on='title')

# Select relevant columns from the merged dataframe
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Display the first few rows of the merged dataframe
movies.head()

# Define a function to convert text containing a list of dictionaries to a list of names
import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Drop rows with missing values and apply the convert function to 'genres' column
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)

# Apply the convert function to 'keywords' column
movies['keywords'] = movies['keywords'].apply(convert)

# Display a sample list of dictionaries
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

# Define a function to extract the first 3 names from a list of dictionaries in 'cast' column
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

# Apply the convert3 function to 'cast' column
movies['cast'] = movies['cast'].apply(convert3)

# Extract the first 3 names from 'cast' column
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])

# Define a function to fetch the director's name from 'crew' column
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Apply the fetch_director function to 'crew' column
movies['crew'] = movies['crew'].apply(fetch_director)

# Display a sample of 5 rows from the dataframe
movies.sample(5)

# Define a function to remove spaces from names in a list
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

# Apply the collapse function to 'cast', 'crew', 'genres', and 'keywords' columns
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Display the first few rows of the modified dataframe
movies.head()

# Tokenize the 'overview' column into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine various columns into a new 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Drop unnecessary columns from the dataframe
new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

# Join the elements in the 'tags' column to form a string
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Import CountVectorizer from scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

# Transform the 'tags' column into a sparse matrix of token counts
vector = cv.fit_transform(new['tags']).toarray()

# Display the shape of the resulting matrix
vector.shape

# Import cosine_similarity from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity matrix
similarity = cosine_similarity(vector)

# Display the similarity matrix
similarity

# Find the index of the movie 'The Lego Movie'
new[new['title'] == 'The Lego Movie'].index[0]

# Define a function to recommend movies based on similarity
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# Recommend movies similar to 'Avatar'
recommend('Avatar')

# Save the modified dataframe and similarity matrix to pickle files
import pickle
pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
