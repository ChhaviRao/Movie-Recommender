import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load data ----------
folder = os.path.dirname(os.path.abspath(__file__))
movies = pd.read_csv(os.path.join(folder, "movies.csv"))
ratings = pd.read_csv(os.path.join(folder, "ratings.csv"))

# ---------- 1. Content-based: movies similar to a movie ----------
movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres_clean"])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_similar(title, n=10):
    matches = movies[movies["title"].str.contains(title, case=False, regex=False)]
    if matches.empty:
        return "No movie found matching '" + title + "'"
    idx = matches.index[0]
    scores = sorted(enumerate(content_sim[idx]), key=lambda x: x[1], reverse=True)
    top = [i for i, _ in scores[1:n + 1]]
    return movies.iloc[top][["title", "genres"]]


# ---------- 2. Collaborative filtering: recommend for a user ----------
user_item = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
user_sim = cosine_similarity(user_item)
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)


def recommend_for_user(user_id, n=10, k=20):
    if user_id not in user_item.index:
        return "Unknown user ID"
    neighbors = user_sim_df[user_id].sort_values(ascending=False)[1:k + 1]
    neighbor_ratings = user_item.loc[neighbors.index]
    weighted = neighbor_ratings.mul(neighbors, axis=0).sum() / neighbors.sum()
    seen = user_item.loc[user_id]
    weighted = weighted.drop(index=seen[seen > 0].index)
    top_ids = weighted.sort_values(ascending=False).head(n).index
    return movies[movies["movieId"].isin(top_ids)][["title", "genres"]]


# ---------- Simple menu ----------
if __name__ == "__main__":
    print("=== Movie Recommender ===")
    print("1. Find movies similar to a movie")
    print("2. Get recommendations for a user")
    choice = input("Choose 1 or 2: ")

    if choice == "1":
        name = input("Enter a movie name (e.g. Toy Story): ")
        print(recommend_similar(name, 10).to_string(index=False))
    elif choice == "2":
        uid = int(input("Enter a user ID (1 to 610): "))
        print(recommend_for_user(uid, 10).to_string(index=False))
    else:
        print("Invalid choice")
