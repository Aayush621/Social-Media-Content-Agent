from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agent import graph, Platform
from typing import List, Optional, Literal  # Add Literal to the import statement
  # Import your existing agent

app = FastAPI(
    title="Content Generation API",
    description="API for generating social media content across multiple platforms",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",  # Common Live Server port
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContentRequest(BaseModel):
    text: str
    platforms: List[Platform] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Sample content to generate",
                "platforms": ["Twitter", "LinkedIn"]
            }
        }

class ContentResponse(BaseModel):
    generated_content: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_content": "Generated content will appear here"
            }
        }

@app.post("/generate-content", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    try:
        result = graph.invoke({
            "text": request.text,
            "platforms": request.platforms
        })
        return ContentResponse(generated_content=result["generated_content"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supported-platforms")
async def get_supported_platforms():
    return {"platforms": ["Twitter", "LinkedIn", "Instagram", "Blog"]} 