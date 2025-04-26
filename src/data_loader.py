import pandas as pd
import os

def load_movielens_data():
    data_dir = r'C:\Users\PC\Desktop\collaborative-filtering-recommender\data'

    # Check if the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")

    # File paths
    ratings_path = os.path.join(data_dir, 'u.data')
    items_path = os.path.join(data_dir, 'u.item')
    users_path = os.path.join(data_dir, 'u.user')

    # Check if all required files exist
    for path in [ratings_path, items_path, users_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")

    # Load rating data with unique column names
    ratings_df = pd.read_csv(ratings_path, sep='\t',
                             names=['user_id_ml', 'item_id_ml', 'rating_ml', 'timestamp_ml'])
    
    # Load item metadata with specific encoding
    items_df = pd.read_csv(items_path, sep='|', encoding='latin-1',
                           names=['item_id_ml', 'title_ml', 'release_date_ml', 'video_release_ml',
                                  'imdb_url_ml'] + [f'genre_{i}_ml' for i in range(19)])
    
    # Load user metadata
    users_df = pd.read_csv(users_path, sep='|',
                           names=['user_id_ml', 'age_ml', 'gender_ml', 'occupation_ml', 'zip_code_ml'])
    
    # Merge dataframes
    full_df = ratings_df.merge(users_df, on='user_id_ml').merge(items_df, on='item_id_ml')
    
    return full_df, ratings_df, users_df, items_df



def preprocess_data(ratings_df):
    # Handle missing values
    ratings_clean = ratings_df.dropna(subset=['user_id_ml', 'item_id_ml', 'rating_ml'])
    
    # Filter out users with too few ratings
    user_rating_counts = ratings_clean['user_id_ml'].value_counts()
    valid_users = user_rating_counts[user_rating_counts >= 5].index
    filtered_ratings = ratings_clean[ratings_clean['user_id_ml'].isin(valid_users)]
    
    return filtered_ratings