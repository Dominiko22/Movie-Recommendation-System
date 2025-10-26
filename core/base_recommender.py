"""
Base Recommender - Core recommendation engine.

Implements content-based filtering using TF-IDF and Cosine Similarity.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender:
    """
    Base class for a content-based recommendation system.

    Uses TF-IDF vectorization and cosine similarity to find
    similar items based on text features (e.g., genres).
    """

    def __init__(self, data_path: str):
        """
        Initialize the recommender.

        Args:
            data_path (str): Path to the CSV file with data.
        """
        self.data_path = data_path
        self.data = None
        self.similarity_matrix = None

        # Load data immediately
        self._load_data()

    def _load_data(self):
        """
        Load dataset from a CSV file and store it in self.data.
        """
        print(f"Loading data from {self.data_path}...")
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.data)} items.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def build_similarity_matrix(self):
        """
        Build a similarity matrix using TF-IDF and Cosine Similarity.

        This is the core of the recommendation algorithm.
        """
        print("Building similarity matrix...")

        # Clean and prepare genres for TF-IDF
        # Example: "Action|Sci-Fi" → "Action Sci-Fi"
        self.data['genres_clean'] = self.data['genres'].str.replace('|', ' ', regex=False)

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.data['genres_clean'])
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Cosine Similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        print("Similarity matrix built successfully.")

    def get_recommendations(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get similar items based on a given title.

        Args:
            title (str): Title of the item to find similar items for.
            top_n (int): Number of recommendations to return.

        Returns:
            pd.DataFrame: DataFrame containing recommended items and similarity scores.
        """
        try:
            # Find index of the movie
            idx = self.data[self.data['title'].str.lower() == title.lower()].index[0]

            # Get similarity scores for this movie
            sim_scores = list(enumerate(self.similarity_matrix[idx]))

            # Sort by similarity (descending)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get top N (excluding itself at position 0)
            sim_scores = sim_scores[1:top_n + 1]

            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]

            # Return recommendations
            recommendations = self.data.iloc[movie_indices].copy()
            recommendations['similarity'] = [i[1] for i in sim_scores]

            return recommendations

        except IndexError:
            print(f"Movie '{title}' not found!")
            return pd.DataFrame()

    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search items by partial title match.

        Args:
            query (str): Search query.
            limit (int): Maximum number of results.

        Returns:
            pd.DataFrame: DataFrame with search results.
        """
        query = query.lower()
        results = self.data[self.data['title'].str.lower().str.contains(query, na=False)]
        return results.head(limit)


# Test if this file is run directly
if __name__ == '__main__':
    print("Testing BaseRecommender...")

    # Test with your data
    rec = BaseRecommender('data/movies/movies.csv')
    rec.build_similarity_matrix()

    # Test search
    print("\n=== Search Test ===")
    results = rec.search('matrix')
    print(results[['title', 'genres']])

    # Test recommendations
    print("\n=== Recommendations Test ===")
    recs = rec.get_recommendations('Matrix, The (1999)', top_n=5)
    print(recs[['title', 'genres', 'similarity']])