import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
import os
import json

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("meryl_streep_movies.csv")
    df['Synopsis'] = df['Synopsis'].fillna('')
    df['Genre'] = df['Genre'].fillna('').str.split('/')
    return df

df = load_data()

# --- Feature engineering ---
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
synopsis_tfidf = vectorizer.fit_transform(df['Synopsis']).toarray()

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['Genre'])

df['IMDb_Rating'] = pd.to_numeric(df['IMDb_Rating'], errors='coerce').fillna(0)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)
numeric_features = df[['IMDb_Rating', 'Year']].values
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

from numpy import hstack
features = hstack([synopsis_tfidf, genre_encoded, numeric_scaled])

# --- Clustering ---
n_clusters = st.slider("Select number of clusters", 2, 8, 4)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features)

# --- Load existing labels if available ---
LABELS_FILE = "cluster_labels.json"
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, 'r') as f:
        cluster_labels = json.load(f)
else:
    cluster_labels = {}

# --- Interactive labeling with saved state ---
st.title("🎯 Interactive Cluster Labeling App")

moods = ['Laugh', 'Cry', 'Escape', 'Think Deeply', 'Other']

for c in df['Cluster'].unique():
    st.markdown(f"### 🎬 Cluster {c} - Review Sample Movies")
    sample_movies = df[df['Cluster'] == c][['Title', 'Synopsis']].sample(3)
    for _, row in sample_movies.iterrows():
        st.write(f"**{row['Title']}**")
        st.write(f"📖 {row['Synopsis'][:300]}...")

    default_label = cluster_labels.get(str(c), 'Other')
    selected_mood = st.selectbox(
        f"Assign mood to Cluster {c}",
        moods,
        index=moods.index(default_label) if default_label in moods else 0,
        key=f"mood_{c}"
    )
    cluster_labels[str(c)] = selected_mood

# --- Apply & Save ---
if st.button("✅ Apply & Save Labels"):
    df['Assigned_Mood'] = df['Cluster'].map(lambda x: cluster_labels.get(str(x), 'Other'))
    st.success("Labels applied! Preview below:")
