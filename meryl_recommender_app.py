import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("meryl_streep_movies.csv", encoding="utf-8")
    df['Title'] = df['Title'].str.strip()
    df["Poster_Filename"] = df["Title"].apply(lambda x: f"{x.replace(':','').replace(' ','').replace('!','').replace('\'','').lower()}.jpg")
    df.columns = df.columns.str.strip()
    df['Genre'] = df['Genre'].fillna('').str.split('/')
    df['Synopsis'] = df['Synopsis'].fillna('')
    df['Awards'] = df['Awards'].fillna('')
    mlb_genre = MultiLabelBinarizer()
    genre_encoded = mlb_genre.fit_transform(df['Genre'])
    df_genre = pd.DataFrame(genre_encoded, columns=mlb_genre.classes_)
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    synopsis_encoded = vectorizer.fit_transform(df['Synopsis'])
    df_synopsis = pd.DataFrame(synopsis_encoded.toarray(), columns=vectorizer.get_feature_names_out())
    df['Has_Awards'] = df['Awards'].apply(lambda x: 1 if x.strip() else 0)
    df['Num_Awards'] = df['Awards'].apply(lambda x: len(x.split(',')) if x else 0)
    df['IMDb_Rating'] = pd.to_numeric(df['IMDb_Rating'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    numeric_features = df[['IMDb_Rating', 'Year', 'Has_Awards', 'Num_Awards']].fillna(0)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)
    df_numeric = pd.DataFrame(numeric_scaled, columns=numeric_features.columns)
    final_features = pd.concat([df_numeric.reset_index(drop=True), df_genre.reset_index(drop=True), df_synopsis.reset_index(drop=True)], axis=1)
    df['Synopsis_Sentiment'] = df['Synopsis'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df, final_features

@st.cache_data
def auto_cluster_and_map(df, features):
    silhouette_scores = []
    range_n = range(3, 7)
    for n in range_n:
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        preds = km.fit_predict(features)
        score = silhouette_score(features, preds)
        silhouette_scores.append(score)
    best_n = range_n[silhouette_scores.index(max(silhouette_scores))]
    st.write(f"📊 Best clusters detected: {best_n} (Silhouette Score: {max(silhouette_scores):.2f})")
    kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features)

    mood_keywords = {
        "laugh": ["comedy", "funny", "satire", "romantic", "light"],
        "cry": ["tragedy", "drama", "emotional", "sad", "heartbreaking"],
        "escape": ["fantasy", "adventure", "musical", "magic", "whimsical"],
        "think deeply": ["thriller", "biography", "political", "historical", "war"]
    }

    cluster_mood = {}
    cluster_mood_scores = {}

    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]
        synopsis_text = " ".join(cluster_df['Synopsis'].tolist()).lower()
        avg_sentiment = cluster_df['Synopsis_Sentiment'].mean()

        mood_scores = {}
        for mood, kws in mood_keywords.items():
            keyword_score = sum(synopsis_text.count(kw) for kw in kws)
            genre_score = cluster_df['Genre'].apply(lambda g: sum(1 for kw in kws if kw in [x.lower() for x in g])).sum()
            if mood == 'laugh':
                sentiment_factor = max(avg_sentiment, 0)
            elif mood == 'cry':
                sentiment_factor = max(-avg_sentiment, 0)
            elif mood == 'escape':
                sentiment_factor = abs(avg_sentiment)
            else:
                sentiment_factor = 1 - abs(avg_sentiment)
            overall_score = (0.5 * keyword_score) + (0.3 * genre_score) + (0.2 * sentiment_factor * 10)
            mood_scores[mood] = overall_score

        cluster_mood_scores[cluster] = mood_scores
        assigned_mood = max(mood_scores, key=mood_scores.get)
        cluster_mood[cluster] = assigned_mood

    return df, cluster_mood, cluster_mood_scores

df, final_features = load_and_preprocess_data()
df, cluster_mood, cluster_mood_scores = auto_cluster_and_map(df, final_features)

# UI
st.title("🎬 Smarter Meryl Streep Movie Recommender")

user_mood = st.selectbox("What's your mood?", ["Laugh", "Cry", "Escape", "Think Deeply"]).lower()
mood_to_cluster = {v: k for k, v in cluster_mood.items()}
chosen_cluster = mood_to_cluster.get(user_mood)

if st.button('Get Smarter Recommendations!'):
    if chosen_cluster is None:
        st.error("Sorry, no matching cluster found. Try another mood.")
    else:
        recommended = df[df['Cluster'] == chosen_cluster]
        suggestions = recommended.sample(3) if len(recommended) >= 3 else recommended
        st.success(f"Movies for your **'{user_mood.capitalize()}'** mood:")
        for idx, row in suggestions.iterrows():
            st.write(f"🎬 **{row['Title']}** ({int(row['Year'])})")
            st.caption(f"Genre: {', '.join(row['Genre'])}")
            poster_filename = row.get("Poster_Filename", "")
            poster_path = f"posters/{poster_filename}"
            if os.path.exists(poster_path):
                st.image(poster_path, width=200)
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)

# Visualize cluster mood confidence
st.header("🔍 Cluster Mood Confidence (Per Cluster)")
for cluster in df['Cluster'].unique():
    mood_scores = cluster_mood_scores[cluster]
    plt.figure(figsize=(6, 3))
    sns.barplot(x=list(mood_scores.keys()), y=list(mood_scores.values()), palette='muted')
    plt.title(f"Cluster {cluster} - Assigned Mood: {cluster_mood[cluster].capitalize()}")
    plt.ylabel("Confidence Score")
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()
