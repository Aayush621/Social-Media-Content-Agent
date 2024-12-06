from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from agent import graph, Platform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContentRequest(BaseModel):
    text: str
    platforms: List[Platform]
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Sample content to generate",
                "platforms": ["Twitter", "LinkedIn"]
            }
        }

class ContentResponse(BaseModel):
    generated_content: Dict[str, str]
    image_prompt: Optional[str] = None
    image_base64: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_content": "Generated content for all platforms",
                "image_prompt": "sunset beach yoga peaceful meditation",
                "image_base64": "data:image/png;base64,..."
            }
        }

@app.post("/generate-content", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    try:
        logger.info(f"Received request for platforms: {request.platforms}")
        
        # Invoke the graph with the request data
        result = graph.invoke({
            "text": request.text,
            "platforms": request.platforms
        })
        
        logger.info("Graph execution completed")
        logger.info(f"Result keys: {result.keys()}")
        
        # Log image generation details
        logger.info(f"Image prompt: {result.get('image_prompt')}")
        logger.info(f"Has image data: {bool(result.get('image_base64'))}")
        
        # Extract platform-specific content
        platform_content = {}
        if isinstance(result.get("generated_content"), dict):
            platform_content = result["generated_content"]
        
        response = ContentResponse(
            generated_content=platform_content,
            image_prompt=result.get("image_prompt"),
            image_base64=result.get("image_base64")
        )
        
        logger.info("Response created successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in generate_content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "An error occurred while generating content"
            }
        )

@app.get("/supported-platforms")
async def get_supported_platforms():
    return {"platforms": ["Twitter", "LinkedIn", "Instagram", "Blog"]} 