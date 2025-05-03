# meryl_recommender_app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import re

# ------------------------------
# Load and preprocess data
# ------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("meryl_streep_movies.csv", encoding="utf-8")
    df['Title'] = df['Title'].str.strip()

# Map movie titles to local image filenames
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
    "Silkwood": "Silkwood3.jpg",  # or "Silkwood9.jpg"
    "Sophie's Choice": "Sophieschoice1.jpg"
}

df["Poster_Filename"] = df["Title"].map(title_to_filename)


    # Preprocessing
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

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(final_features)

    return df


df = load_data()

# ------------------------------
# Streamlit User Interface
# ------------------------------

st.title("🎬 Meryl Streep Movie Recommender")
st.write("Pick your mood and we'll recommend some Meryl Streep classics!")

# Define moods
moods = ["Laugh", "Cry", "Escape", "Think Deeply"]

# Mood selector
user_mood = st.selectbox("What mood are you in?", moods)

# Mood to cluster mapping
mood_to_cluster = {
    "laugh": 0,
    "cry": 1,
    "escape": 2,
    "think deeply": 3
}

# Button to get recommendation
if st.button('Get Recommendations!'):
    user_mood_lower = user_mood.lower().strip()
    chosen_cluster = mood_to_cluster.get(user_mood_lower)

    if chosen_cluster is None:
        st.error("Sorry, we couldn't match your mood.")
    else:
        recommended_movies = df[df['Cluster'] == chosen_cluster]
        suggestions = recommended_movies.sample(3) if len(recommended_movies) >= 3 else recommended_movies

        st.success("Here are your recommended movies!")
        
        for idx, row in suggestions.iterrows():
            st.write(f"🎬 **{row['Title']}** ({int(row['Year'])})")
            st.caption(f"Genre: {', '.join(row['Genre'])}")


            poster_filename = row.get("Poster_Filename", "")
            poster_path = f"posters/{poster_filename}"
    
            if os.path.exists(poster_path):
                st.image(poster_path, width=200)
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)

