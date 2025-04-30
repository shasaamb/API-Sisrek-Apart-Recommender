# 🏠 Apartment Recommendation System API

This project is a **Content-Based Recommendation API** tailored for the apartment rental market in Jakarta. It enables personalized apartment suggestions by processing structured user preferences through a hybrid form, returning results based on TF-IDF similarity between user input and apartment metadata.

## 🚀 Features

- ✅ Form-based apartment recommendation
- ✅ TF-IDF vectorization of apartment features
- ✅ Cosine similarity for preference matching
- ✅ Filtering based on price, size, rating
- ✅ Clean API output including apartment name, price, rating, and key metadata
- ✅ Deployable via FastAPI and ngrok

---

## 🛠️ Tech Stack

- Python
- FastAPI
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- Pandas
- Uvicorn
- ngrok (for public testing)

---

## 🔧 Project Structure

- `app.py`: Main Flask API to serve recommendations
- `utils/hybrid_model.py`: Core logic for hybrid filtering
- `data/`: CSV apartment listings and reviews
- `models/`: Pre-trained models in .pkl format
- `form_illustration/`: UI wireframe references

---

## 🚀 Running Locally

1. Clone this repo
2. Install dependencies:

```bash
pip install -r requirements.txt
