from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_rmse(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

def calculate_precision_recall_at_k(true_items, predicted_items, k=10):
    relevant_items = set(true_items)
    top_k_predicted = set(predicted_items[:k])
    
    tp = len(relevant_items.intersection(top_k_predicted))
    precision = tp / k if k > 0 else 0
    recall = tp / len(relevant_items) if len(relevant_items) > 0 else 0
    
    return precision, recall