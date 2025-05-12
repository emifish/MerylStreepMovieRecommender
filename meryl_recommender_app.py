import streamlit as st
import pandas as pd
import os
import re
from textblob import TextBlob
from fuzzywuzzy import process, fuzz
from PIL import Image, UnidentifiedImageError

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("meryl_streep_movies.csv")
    df['Synopsis'] = df['Synopsis'].fillna('')
    df['Genre'] = df['Genre'].fillna('').str.split('/')
    df['Synopsis_Sentiment'] = df['Synopsis'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

df = load_data()

# --- Editable Heuristic Rules UI ---
st.sidebar.header("⚙️ Heuristic Mood Rules Configuration")

laugh_keywords = st.sidebar.text_area("Laugh keywords (comma separated)", "comedy, funny, satire, romantic, light").lower().split(',')
cry_keywords = st.sidebar.text_area("Cry keywords (comma separated)", "tragedy, drama, emotional, sad, heartbreaking").lower().split(',')
escape_keywords = st.sidebar.text_area("Escape keywords (comma separated)", "fantasy, adventure, musical, magic, whimsical").lower().split(',')
think_keywords = st.sidebar.text_area("Think Deeply keywords (comma separated)", "thriller, biography, political, historical, war").lower().split(',')

laugh_sentiment = st.sidebar.number_input("Laugh sentiment threshold", value=0.3)
cry_sentiment = st.sidebar.number_input("Cry sentiment threshold", value=-0.2)

# --- Mood assignment logic ---
def assign_mood(row):
    synopsis = row['Synopsis'].lower()
    genres = [g.lower() for g in row['Genre']]
    sentiment = row['Synopsis_Sentiment']

    if any(g in genres for g in laugh_keywords) or any(kw.strip() in synopsis for kw in laugh_keywords) or sentiment > laugh_sentiment:
        return 'Laugh'
    elif any(g in genres for g in cry_keywords) and (any(kw.strip() in synopsis for kw in cry_keywords) or sentiment < cry_sentiment):
        return 'Cry'
    elif any(g in genres for g in escape_keywords) or any(kw.strip() in synopsis for kw in escape_keywords):
        return 'Escape'
    else:
        return 'Think Deeply'

df['Assigned_Mood'] = df.apply(assign_mood, axis=1)

# --- Poster fuzzy matching ---
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

def safe_display_poster(title):
    try:
        poster_filename = match_poster(title)
        poster_path = f"{POSTERS_FOLDER}{poster_filename}"
        if poster_filename and os.path.exists(poster_path):
            with Image.open(poster_path) as img:
                st.image(poster_path, width=200)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
    except (UnidentifiedImageError, OSError, FileNotFoundError):
        st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)

# --- UI ---
st.title("🎬 Meryl Streep Movie Recommender (Heuristic + Fuzzy Posters)")

user_mood = st.selectbox("What mood are you in?", ['Laugh', 'Cry', 'Escape', 'Think Deeply'])

recommended = df[df['Assigned_Mood'].str.lower() == user_mood.lower()]

if recommended.empty:
    st.warning(f"No movies found for mood '{user_mood}'.")
else:
    st.success(f"Movies for your **'{user_mood}'** mood:")
    suggestions = recommended.sample(3) if len(recommended) >= 3 else recommended
    for _, row in suggestions.iterrows():
        st.write(f"🎬 **{row['Title']}** ({int(row['Year'])})")
        st.caption(f"Genre: {', '.join(row['Genre'])}")
        st.write(f"📖 {row['Synopsis'][:300]}...")

        # Use safe display poster function
        safe_display_poster(row['Title'])

# --- Preview table ---
st.header("📋 All Movies with Assigned Moods (Heuristic - User Tuned)")

selected_mood_filter = st.selectbox("Filter table by mood", ['All'] + df['Assigned_Mood'].unique().tolist())

if selected_mood_filter != 'All':
    st.dataframe(df[df['Assigned_Mood'] == selected_mood_filter][['Title', 'Year', 'Genre', 'Assigned_Mood', 'Synopsis_Sentiment']])
else:
    st.dataframe(df[['Title', 'Year', 'Genre', 'Assigned_Mood', 'Synopsis_Sentiment']])
