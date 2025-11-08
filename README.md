# 🎬 Movie Recommendation System (Content-Based Filtering)

A Python-based **Movie Recommendation System** that suggests movies similar to the one entered by the user.  
It compares movies based on **keywords, cast, director, and genres**, and finds the most similar movies using **CountVectorizer** and **Cosine Similarity**.

---

## 📌 Description (Short)

This program takes a movie title as input and returns a ranked list of similar movies.  
It uses a **content-based filtering** technique, which recommends movies using their metadata (not user ratings).

->

## 🚀 Features

- Suggests movies based on similarity of movie content
- Uses NLP (natural language processing) to convert movie text to vectors
- No APIs needed — fully offline
- Beginner-friendly and easy to modify

---

## 🧠 How It Works

1. Load movie dataset from `movie_dataset.csv`
2. Extract and combine text-based features:
   - `keywords`
   - `cast`
   - `genres`
   - `director`
3. Convert combined text into vectors with `CountVectorizer`
4. Calculate similarity between movies using **Cosine Similarity**
5. Display the top similar movies

---

