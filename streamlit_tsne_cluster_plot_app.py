import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎬 Meryl Streep Movie Clustering - t-SNE Visualization")

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

# Combine features
from numpy import hstack
features = hstack([synopsis_tfidf, genre_encoded, numeric_scaled])

# --- Clustering ---
n_clusters = st.slider("Select number of clusters", 2, 8, 4)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features)

# --- 2D reduction with t-SNE ---
st.info("⚡ Running t-SNE (can take a few seconds)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1000)
features_2d = tsne.fit_transform(features)
df['TSNE1'] = features_2d[:, 0]
df['TSNE2'] = features_2d[:, 1]

# --- Plot inside Streamlit ---
st.subheader("📊 2D t-SNE Cluster Plot")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='TSNE1', y='TSNE2', hue='Cluster', palette='Set2', s=100)
for i, row in df.iterrows():
    plt.text(row['TSNE1'] + 1, row['TSNE2'] + 1, row['Title'], fontsize=7)
plt.title("2D t-SNE Cluster Plot of Movies")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title='Cluster')
plt.grid(True)

st.pyplot(plt)
plt.close()
