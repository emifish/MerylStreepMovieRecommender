import streamlit as st
import pandas as pd
import os
import re
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from fuzzywuzzy import process, fuzz
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Load data ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("meryl_streep_movies.csv")
    df['Synopsis'] = df['Synopsis'].fillna('')
    df['Genre'] = df['Genre'].fillna('').str.split('/')
    df['Awards'] = df['Awards'].fillna('')
    df['Synopsis_Sentiment'] = df['Synopsis'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

df = load_and_preprocess_data()

# --- Feature Engineering ---
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
synopsis_tfidf = vectorizer.fit_transform(df['Synopsis']).toarray()

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['Genre'])

df['IMDb_Rating'] = pd.to_numeric(df['IMDb_Rating'], errors='coerce').fillna(0)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)
numeric_features = df[['IMDb_Rating', 'Year']].values
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

features = np.hstack([synopsis_tfidf, genre_encoded, numeric_scaled])

# --- HDBSCAN clustering ---
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
df['Cluster'] = clusterer.fit_predict(features)
df['Cluster'] = df['Cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')

# --- Mood scoring logic ---
mood_keywords = {
    "laugh": ["comedy", "funny", "satire", "romantic", "light"],
    "cry": ["tragedy", "drama", "emotional", "sad", "heartbreaking"],
    "escape": ["fantasy", "adventure", "musical", "magic", "whimsical"],
    "think deeply": ["thriller", "biography", "political", "historical", "war"]
}

cluster_mood = {}
cluster_mood_scores = {}

for cluster in df['Cluster'].unique():
    if cluster == 'Noise':
        continue
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

# --- Poster fuzzy matching logic ---
POSTERS_FOLDER = "posters/"
poster_files = [f for f in os.listdir(POSTERS_FOLDER) if f.lower().endswith(('.jpg', '.png'))]
poster_file_keys = [os.path.splitext(f)[0] for f in poster_files]

def clean_text(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def match_poster(title):
    cleaned_title = clean_text(title)
    matches = process.extractOne(cleaned_title, [clean_text(f) for f in poster_file_keys], scorer=fuzz.partial_ratio)
    if matches and matches[1] >= 80:
        matched_index = [clean_text(f) for f in poster_file_keys].index(matches[0])
        return poster_files[matched_index]
    return None

# --- UI ---
st.title("🎬 Meryl Streep Movie Recommender (with HDBSCAN & Fuzzy Posters)")

user_mood = st.selectbox("What's your mood?", ['Laugh', 'Cry', 'Escape', 'Think Deeply'])

# Map mood to detected clusters
mood_to_clusters = [c for c, m in cluster_mood.items() if m.lower() == user_mood.lower()]
recommended = df[df['Cluster'].isin(mood_to_clusters)]

if recommended.empty:
    st.warning(f"No movies found for mood '{user_mood}'.")
else:
    st.success(f"Movies for your **'{user_mood}'** mood:")
    suggestions = recommended.sample(3) if len(recommended) >= 3 else recommended
    for _, row in suggestions.iterrows():
        st.write(f"🎬 **{row['Title']}** ({int(row['Year'])})")
        st.caption(f"Genre: {', '.join(row['Genre'])}")
        st.write(f"📖 {row['Synopsis'][:300]}...")
        
        try:
            poster_filename = match_poster(row['Title'])
            poster_path = f"{POSTERS_FOLDER}{poster_filename}"
            if poster_filename and os.path.exists(poster_path):
                with Image.open(poster_path) as img:
                    st.image(poster_path, width=200)
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
        

# --- Visualize cluster mood confidence ---
    # #st.header("🔍 Cluster Mood Confidence (Per Cluster)")
    # for cluster in df['Cluster'].unique():
    #     if cluster == 'Noise':
    #         continue
    #     mood_scores = cluster_mood_scores[cluster]
    #     plt.figure(figsize=(6, 3))
    #     sns.barplot(x=list(mood_scores.keys()), y=list(mood_scores.values()), palette='muted')
    #     plt.title(f"{cluster} - Assigned Mood: {cluster_mood[cluster].capitalize()}")
    #     plt.ylabel("Confidence Score")
    #     plt.xticks(rotation=45)
    #     st.pyplot(plt)
    #     plt.close()
