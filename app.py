from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
class Facilities(BaseModel):
    furniture: Optional[List[str]] = []
    kitchen: Optional[List[str]] = []
    bathroom: Optional[List[str]] = []
    utility: Optional[List[str]] = []

class UserForm(BaseModel):
    tipe_lokasi: List[str]
    tipe_kamar_tidur: str
    facilities: Facilities
    descriptions_proximity_category: List[str]
    descriptions_building_facility: List[str]

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
    top_n: Optional[int] = 10
    
# --- Helper functions ---
def build_query(form: UserForm) -> str:
    query = []
    query += form.tipe_lokasi
    query += form.tipe_kamar_tidur.split("_")
    for group in form.facilities.__dict__.values():
        query += group
    query += form.descriptions_proximity_category
    query += form.descriptions_building_facility
    return " ".join(query)

def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    return df[
        (df["apart_price"] >= f.price_range_min) &
        (df["apart_price"] <= f.price_range_max) &
        (df["apart_rating"] >= f.rating_range_min) &
        (df["apart_rating"] <= f.rating_range_max) &
        (df["apart_ukuran"] >= f.size_range_min) &
        (df["apart_ukuran"] <= f.size_range_max)
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

        # Normalize to 1-5 scale
        cbf_min = scored_df["cbf_score"].min()
        cbf_max = scored_df["cbf_score"].max()
        scored_df["cbf_score_scaled"] = (scored_df["cbf_score"] - cbf_min) / (cbf_max - cbf_min + 1e-9) * 4 + 1

        # Convert to percentage similarity
        scored_df["similarity_percent"] = scored_df["cbf_score"] * 100

        filtered_df = apply_filters(scored_df, req.filters)

        result = filtered_df.sort_values(by="cbf_score", ascending=False).head(req.top_n)[
            [
                "id", "apart_name", "images", "detail_url", "descriptions",
                "apart_owner", "apart_owner_link", "apart_owner_verified",
                "apart_location", "apart_address_og", "apart_address_cleaned",
                "apart_price", "apart_rating", "apart_ukuran",
                "tipe_apart", "tipe_kamar_tidur", "total_kamar_tidur", "total_kamar_mandi",
                "facilities", "description_proximity", "description_proximity_category",
                "description_building_facilities", "bed_config", "lokasi_token",
                "facilities_token", "proximity_token", "building_facility_token",
                "cbf_feature_string", "bedroom_token", "bathroom_token", "token"
            ]
        ]

        return result.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def root():
    return {"message": "API is up and running!"}

# --- Launch server locally (for dev testing only) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
