import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import streamlit as st

# Load movies data
movies_data = pd.read_csv('movies.csv')

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace null values with empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine all selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + \
                    movies_data['director']

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute cosine similarity
similarity = cosine_similarity(feature_vectors)

# Function to recommend movies
def recommend_movies(movie_name, movies_data, similarity):
    # List of all movie names
    list_of_all_titles = movies_data['title'].tolist()
    
    # Finding close match
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0] if find_close_match else None
    
    if close_match:
        # Find index of movie
        index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        
        # Getting a list of similar movies
        similarity_score = list(enumerate(similarity[index_of_movie]))
        
        # Sorting based on similarity score
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Return top recommended movies
        return [movies_data.iloc[movie[0]]['title'] for movie in sorted_similar_movies[:30]]
    else:
        return []

# Main function to create the Streamlit app
def main():
    st.title('Movie Recommendation System')
    
    # User input for movie name
    movie_name = st.text_input('Enter your favorite movie name:')
    
    # Button to trigger recommendation
    if st.button('Recommend'):
        # Call recommendation function
        recommended_movies = recommend_movies(movie_name, movies_data, similarity)
        
        # Display recommended movies
        st.subheader('Movies suggested for you:')
        if recommended_movies:
            for i, movie in enumerate(recommended_movies, start=1):
                st.write(f'{i}. {movie}')
        else:
            st.write('No similar movies found.')

if __name__ == "__main__":
    main()
