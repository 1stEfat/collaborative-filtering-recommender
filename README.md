# 🎬 Movie Recommendation Engine

A production-ready collaborative filtering system implementing both user-based and item-based approaches on the MovieLens 100K dataset.

![Recommendation Visualization](https://example.com/path/to/demo.gif) *(Example visualization placeholder)*

## ✨ Features

- **Dual Algorithm Implementation**
  - User-Based Collaborative Filtering
  - Item-Based Collaborative Filtering
- **Advanced Evaluation**
  - RMSE, Precision@K, Recall@K metrics
  - Similarity matrix visualizations
- **Production-Grade Outputs**
  - Top-N recommendations in CSV format
  - JSON evaluation reports
- **Modular Design**
  - Clean separation of data, models, and evaluation
  - Configurable hyperparameters

## 🛠️ Tech Stack

| Category       | Technologies                          |
|----------------|---------------------------------------|
| Core           | Python 3.9+, NumPy, pandas            |
| ML             | scikit-learn, Surprise                |
| Visualization  | Matplotlib, Seaborn, Plotly           |
| Infrastructure | Jupyter, pipenv                       |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or pipenv

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Install dependencies (using pipenv)
pipenv install --dev
pipenv shell

# Or with pip
pip install -r requirements.txt
