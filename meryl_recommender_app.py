import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import random
import re
import os

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("meryl_streep_movies.csv", encoding="utf-8")
    df['Title'] = df['Title'].str.strip()

    title_to_filename = {
        "August: Osage County": "Augustosagecounty2.jpg",
        "Death Becomes Her": "Deathbecomesher2.jpg",
        "The Deer Hunter": "Deerhunter4.jpg",
        "The Devil Wears Prada": "Devilwearsprada1.jpg",
        "Don't Look Up": "Dontlookup1.jpg",
        "Doubt": "Doubt1.jpg",
        "Falling in Love": "Fallinginlove2.jpg",
        "Florence Foster Jenkins": "Florencefosterjenkins1.jpg",
        "The French Lieutenant's Woman": "Frenchlieutenantswoman2.jpg",
        "Heartburn": "Heartburn1.jpg",
        "The Hours": "Hours.jpg",
        "Into the Woods": "Intothewoods7.jpg",
        "The Iron Lady": "Ironlady2.jpg",
        "Ironweed": "Ironweed1.jpg",
        "It's Complicated": "Itscomplicated2.jpg",
        "Julia": "Julia1.jpg",
        "Kramer vs. Kramer": "Kramervskramer2.jpg",
        "Lions for Lambs": "Lionsforlambs1.jpg",
        "Little Women": "Littlewomen20195.jpg",
        "Mamma Mia!": "Mammamia21.jpg",
        "The Manchurian Candidate": "Manchuriancandidate.jpg",
        "Manhattan": "Manhattan6.jpg",
        "Mary Poppins Returns": "Marypoppinsreturns2.jpg",
        "Out of Africa": "Outofafrica1.jpg",
        "The Post": "Post2.jpg",
        "Postcards from the Edge": "Postcardsfromtheedge1.jpg",
        "Silkwood": "Silkwood3.jpg",
        "Sophie's Choice": "Sophieschoice1.jpg"
    }

    df["Poster_Filename"] = df["Title"].map(title_to_filename)

    df.columns = df.columns.str.strip()
    df['Genre'] = df['Genre'].fillna('').str.split('/')
    df['Mood_Tags'] = df['Mood_Tags'].fillna('').str.split(', ')
    df['Synopsis'] = df['Synopsis'].fillna('')
    df['Awards'] = df['Awards'].fillna('')

    mlb_genre = MultiLabelBinarizer()
    mlb_mood = MultiLabelBinarizer()

    genre_encoded = mlb_genre.fit_transform(df['Genre'])
    mood_encoded = mlb_mood.fit_transform(df['Mood_Tags'])

    df_genre = pd.DataFrame(genre_encoded, columns=mlb_genre.classes_)
    df_mood = pd.DataFrame(mood_encoded, columns=mlb_mood.classes_)

    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    synopsis_encoded = vectorizer.fit_transform(df['Synopsis'])
    df_synopsis = pd.DataFrame(synopsis_encoded.toarray(), columns=vectorizer.get_feature_names_out())

    df['Has_Awards'] = df['Awards'].apply(lambda x: 1 if x.strip() else 0)
    df['Num_Awards'] = df['Awards'].apply(lambda x: len(x.split(',')) if x else 0)
    df['Num_Nominations'] = df['Awards'].str.count('Nomination', flags=re.IGNORECASE)
    df['Num_Wins'] = df['Awards'].str.count('Win', flags=re.IGNORECASE)
    df[['Num_Nominations', 'Num_Wins']] = df[['Num_Nominations', 'Num_Wins']].fillna(0)

    df['IMDb_Rating'] = pd.to_numeric(df['IMDb_Rating'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    numeric_features = df[['IMDb_Rating', 'Year', 'Has_Awards', 'Num_Awards', 'Num_Nominations', 'Num_Wins']].fillna(0)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)
    df_numeric = pd.DataFrame(numeric_scaled, columns=numeric_features.columns)

    final_features = pd.concat([
        df_numeric.reset_index(drop=True),
        df_genre.reset_index(drop=True),
        df_mood.reset_index(drop=True),
        df_synopsis.reset_index(drop=True)
    ], axis=1)

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
    st.write(f"📊 Auto-detected best clusters: {best_n} (Silhouette Score: {max(silhouette_scores):.2f})")

    kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(features)

    mood_keywords = {
        "laugh": ["comedy", "funny", "satire", "romantic"],
        "cry": ["tragedy", "drama", "emotional", "heartbreaking"],
        "escape": ["fantasy", "adventure", "musical", "magic"],
        "think deeply": ["thriller", "biography", "political", "historical", "war"]
    }

    cluster_mood = {}
    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]
        synopsis_text = " ".join(cluster_df['Synopsis'].tolist()).lower()
        mood_scores = {mood: sum(synopsis_text.count(kw) for kw in kws) for mood, kws in mood_keywords.items()}
        assigned_mood = max(mood_scores, key=mood_scores.get)
        cluster_mood[cluster] = assigned_mood

    return df, cluster_mood

df, final_features = load_and_preprocess_data()
df, cluster_mood = auto_cluster_and_map(df, final_features)

# UI
st.title("🎬 Meryl Streep Movie Recommender (Dynamic Clusters)")

user_mood = st.selectbox("What's your mood?", ["Laugh", "Cry", "Escape", "Think Deeply"]).lower()

cluster_to_mood_inverse = {v: k for k, v in cluster_mood.items()}
chosen_cluster = cluster_to_mood_inverse.get(user_mood)

if st.button('Get Recommendations!'):
    if chosen_cluster is None:
        st.error("No matching cluster found for this mood. Try another mood.")
    else:
        recommended = df[df['Cluster'] == chosen_cluster]
        suggestions = recommended.sample(3) if len(recommended) >= 3 else recommended

        st.success(f"Movies for your '{user_mood.capitalize()}' mood:")
        for idx, row in suggestions.iterrows():
            st.write(f"🎬 **{row['Title']}** ({int(row['Year'])})")
            st.caption(f"Genre: {', '.join(row['Genre'])}")
            poster_filename = row.get("Poster_Filename", "")
            poster_path = f"posters/{poster_filename}"
            if os.path.exists(poster_path):
                st.image(poster_path, width=200)
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
