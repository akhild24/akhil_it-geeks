from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Semantic Group Chat Search API", version="1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only, should be specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

@app.get("/")
def read_root():
    return {"message": "Welcome to Semantic Group Chat Search API"}
