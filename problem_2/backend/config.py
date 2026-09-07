import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Semantic Group Chat Search"
    api_v1_str: str = "/api/v1"
    
    # Model configs
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()
