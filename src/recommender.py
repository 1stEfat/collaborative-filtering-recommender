import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF:
    def __init__(self, min_common_items=5):
        self.min_common = min_common_items
        self.user_similarity_matrix_ubcf = None
        self.ratings_matrix_ubcf = None
        return self
        
    def fit(self, ratings_matrix):
        self.ratings_matrix_ubcf = ratings_matrix.fillna(0)
        self.user_similarity_matrix_ubcf = cosine_similarity(self.ratings_matrix_ubcf)
        
    def predict_rating_ubcf(self, user_id, item_id, top_n=10):
        user_idx = user_id - 1  # Assuming user IDs start at 1
        item_ratings = self.ratings_matrix_ubcf.iloc[:, item_id-1]
        
        similar_users = np.argsort(self.user_similarity_matrix_ubcf[user_idx])[::-1][1:top_n+1]
        similar_ratings = item_ratings.iloc[similar_users]
        
        valid_ratings = similar_ratings[similar_ratings > 0]
        if len(valid_ratings) < self.min_common:
            return np.nan
        return valid_ratings.mean()
    def generate_topn_ubcf(self,model, user_id, user_item_matrix, n=10):
        """Generate top-N recommendations using User-Based CF"""
        user_rated = user_item_matrix.loc[user_id].dropna().index
        all_items = user_item_matrix.columns.astype(int)
        unrated = [i for i in all_items if i not in user_rated]
        
        preds = []
        for item_id in unrated:
            pred = model.predict_rating_ubcf(user_id, item_id)
            if not np.isnan(pred):
                preds.append((item_id, pred))
        
        return sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    


class ItemBasedCF:
    def __init__(self, min_common_users=5):
        self.min_common = min_common_users
        self.item_similarity_matrix_ibcf = None
        self.ratings_matrix_ibcf = None
        
    def fit(self, ratings_matrix):
        self.ratings_matrix_ibcf = ratings_matrix.fillna(0)
        self.item_similarity_matrix_ibcf = cosine_similarity(self.ratings_matrix_ibcf.T)
        return self
        
    def predict_rating_ibcf(self, user_id, item_id, top_n=10):
        item_idx = item_id - 1
        user_ratings = self.ratings_matrix_ibcf.iloc[user_id-1]
        
        similar_items = np.argsort(self.item_similarity_matrix_ibcf[item_idx])[::-1][1:top_n+1]
        similar_ratings = user_ratings.iloc[similar_items]
        
        valid_ratings = similar_ratings[similar_ratings > 0]
        if len(valid_ratings) < self.min_common:
            return np.nan
        return valid_ratings.mean()
    def generate_topn_ibcf(self,model, user_id, user_item_matrix, n=10):
        """Generate top-N recommendations using Item-Based CF"""
        user_ratings = user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings.isna()].index
        
        preds = []
        for item_id in unrated_items:
            pred = model.predict_rating_ibcf(user_id, item_id)
            if not np.isnan(pred):
                preds.append((item_id, pred))
        
        return sorted(preds, key=lambda x: x[1], reverse=True)[:n]