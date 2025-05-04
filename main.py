from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
from typing import Optional
import logging
import re
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Marketing Assistant",
    description="An AI-powered marketing assistant using Hugging Face models",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Hugging Face InferenceClient
MODEL_ID = "distilgpt2"
token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

hf_client = InferenceClient(model=MODEL_ID, token=token)

# Improved ad prompt template for gpt2 with more diverse examples, creative slogans, and a delimiter
def build_ad_prompt(product_description, add_theme):
    return (
        "Product: Running shoes\n"
        "Theme: Sporty\n"
        "Ad Copy: Step into comfort and style with our new running shoes, designed for champions. Perfect for every stride.\n\n"
        "Product: Organic face cream\n"
        "Theme: Natural beauty\n"
        "Ad Copy: Reveal your natural glow with our organic face cream. Pure ingredients for radiant, healthy skin.\n\n"
        "Product: Female underwear\n"
        "Theme: Classy\n"
        "Ad Copy: Discover elegance and comfort with our premium female underwear collection. Designed to make you feel confident and classy every day.\n\n"
        "Product: Shapewear\n"
        "Theme: Body positivity\n"
        "Ad Copy: Introducing our new shapewear line that celebrates every curve. Feel confident and comfortable in your own skin.\n\n"
        "Product: Lingerie\n"
        "Theme: Confidence\n"
        "Ad Copy: Unleash your inner confidence with our luxurious lingerie. Perfect for every occasion, made for every woman.\n\n"
        "Product: Silk robe\n"
        "Theme: Luxury\n"
        "Ad Copy: Wrap yourself in pure luxury with our silk robes. The perfect blend of comfort and sophistication.\n\n"
        "Product: Eco-friendly water bottle\n"
        "Theme: Sustainability\n"
        "Ad Copy: Stay hydrated and save the planet, one sip at a time.\n\n"
        "Product: Smartwatch\n"
        "Theme: Innovation\n"
        "Ad Copy: Stay connected, track your health, and never miss a beat.\n\n"
        "Product: Yoga mat\n"
        "Theme: Wellness\n"
        "Ad Copy: Find your balance and flow with our premium yoga mats.\n\n"
        "Product: Vegan chocolate\n"
        "Theme: Indulgence\n"
        "Ad Copy: Satisfy your sweet tooth with our rich, dairy-free vegan chocolate.\n\n"
        f"Product: {product_description}\n"
        f"Theme: {add_theme}\n"
        "Ad Copy:"
        "\n### END"
    )

class MarketingRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 500
    temperature: Optional[float] = 0.7

class AdInput(BaseModel):
    product_description: str
    add_theme: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_ready": True
    }

@app.post("/generate")
async def generate_content(request: MarketingRequest):
    """Generate general content based on the provided prompt"""
    try:
        max_tokens = request.max_length or 500
        temperature = request.temperature or 0.7
        logger.info(f"[General] Prompt: {request.prompt}")
        response = hf_client.text_generation(
            prompt=request.prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        generated = response.generated_text if hasattr(response, 'generated_text') else response
        logger.info(f"[General] Response: {generated}")
        return {
            "status": "success",
            "generated_content": generated,
            "parameters": {
                "max_length": max_tokens,
                "temperature": temperature
            }
        }
    except Exception as e:
        logger.error(f"[General] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_ad")
async def generate_ad(input: AdInput):
    """Generate marketing ad copy based on product description and theme"""
    try:
        product = input.product_description.strip()
        theme = input.add_theme.strip()
        logger.info(f"[Ad] Prompt variables: {{'product_description': '{product}', 'add_theme': '{theme}'}}")
        prompt = build_ad_prompt(product, theme)
        logger.info(f"\n[Ad] ===== PROMPT SENT TO MODEL =====\n{prompt}\n[Ad] ===== END PROMPT =====\n")
        response = hf_client.text_generation(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
            stop=["### END"]
        )
        generated = response.generated_text if hasattr(response, 'generated_text') else response
        logger.info(f"[Ad] Raw Model Response (full): {repr(generated)}")
        # Return everything before the delimiter as the ad
        ad_text = generated.split('### END')[0].strip()
        logger.info(f"[Ad] Final ad_text sent to frontend: {repr(ad_text)}")
        return {
            "status": "success",
            "ad": ad_text
        }
    except Exception as e:
        logger.error(f"[Ad] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 