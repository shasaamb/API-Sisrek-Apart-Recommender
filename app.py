from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
import requests
import uvicorn


# --- Init FastAPI ---
app = FastAPI()

# --- Load data and models ---
try:
    url = "https://ruanghuni.thisarcid.com/api/apartments"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    apart_df = pd.DataFrame(response.json())
except Exception as e:
      raise Exception(f"Error fetching apartment data: {str(e)}")

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(apart_df["token"])

# import pickle
# with open("models/tfidf.pkl", "rb") as f:
#     tfidf = pickle.load(f)
# with open("models/tfidf_matrix.pkl", "rb") as f:
#     tfidf_matrix = pickle.load(f)

# --- Pydantic Models ---
class UserFormFacilities(BaseModel):
    furniture: Optional[List[str]] = []
    appliances: Optional[List[str]] = []
    bathroom_features: Optional[List[str]] = []
    conveniences: Optional[List[str]] = []

class UserForm(BaseModel):
    tipe_lokasi: List[str]
    jumlah_kamar_tidur: int
    tipe_kamar_tidur: Optional[List[str]] = []
    facilities: UserFormFacilities
    descriptions_proximity_category: Optional[List[str]] = []
    descriptions_building_facility: Optional[List[str]] = []

class Filters(BaseModel):
    min_price: float
    max_price: float
    min_rating: float
    max_rating: float
    min_size: float
    max_size: float

class RecommendationRequest(BaseModel):
    user_form: UserForm
    filters: Filters
    top_n: int = 10

# --- Helper functions ---
def build_query(form: UserForm) -> str:
    query = []

    query += form.tipe_lokasi

    if form.tipe_kamar_tidur:
        query += form.tipe_kamar_tidur

    facilities_dict = form.facilities.__dict__
    for group in facilities_dict.values():
        if group:  
            query += group

    query += form.descriptions_proximity_category

    query += form.descriptions_building_facility

    return " ".join(query)


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    df = df.copy() 

    df["apart_price"] = pd.to_numeric(df["apart_price"], errors="coerce")
    df["apart_rating"] = pd.to_numeric(df["apart_rating"], errors="coerce")
    df["apart_ukuran"] = pd.to_numeric(df["apart_ukuran"], errors="coerce")

    return df[
        (df["apart_price"] >= f.min_price) &
        (df["apart_price"] <= f.max_price) &
        (df["apart_rating"] >= f.min_rating) &
        (df["apart_rating"] <= f.max_rating) &
        (df["apart_ukuran"] >= f.min_size) &
        (df["apart_ukuran"] <= f.max_size)
    ]

# --- API Endpoint ---
@app.post("/recommendations")
def recommend(req: RecommendationRequest):
    try:
        query_string = build_query(req.user_form)
        query_vec = tfidf.transform([query_string])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        scored_df = apart_df.copy()
        scored_df["cbf_score"] = sim_scores

        cbf_min = scored_df["cbf_score"].min()
        cbf_max = scored_df["cbf_score"].max()
        scored_df["cbf_score_scaled"] = (scored_df["cbf_score"] - cbf_min) / (cbf_max - cbf_min + 1e-9) * 4 + 1

        scored_df["similarity_percent"] = scored_df["cbf_score"] * 100

        filtered_df = apply_filters(scored_df, req.filters)

        result = filtered_df.sort_values(by="cbf_score", ascending=False).head(req.top_n)[
            [
                "id", "cbf_score_scaled"
            ]
        ]

        final_data = result.to_dict(orient="records")

        return {
            "status": "success",
            "recommendations": final_data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "error": str(e)
            }
        )
    
@app.get("/")
def root():
    return {"message": "API is up and running!"}

# --- Launch server locally (for dev testing only) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
