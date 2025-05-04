from dotenv import load_dotenv
import os
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import HfApi

def test_token_loading():
    """Test if token is loaded from .env file"""
    load_dotenv()
    token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    
    if token:
        print("✅ Token loaded from .env file successfully!")
        print(f"Token length: {len(token)} characters")
        return token
    else:
        print("❌ Token not found in .env file!")
        return None

def test_huggingface_api(token):
    """Test token with HuggingFace API"""
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        print("\n✅ HuggingFace API test successful!")
        print(f"Username: {user_info['name']}")
        return True
    except Exception as e:
        print(f"\n❌ HuggingFace API test failed: {str(e)}")
        return False

def test_langchain(token):
    """Test token with LangChain"""
    try:
        # Initialize the HuggingFaceHub LLM with a text generation model
        llm = HuggingFaceHub(
            repo_id="gpt2",  # Using GPT-2 which supports text generation
            huggingfacehub_api_token=token,
            model_kwargs={"temperature": 0.7, "max_length": 50}
        )
        
        # Test with a simple prompt
        response = llm.invoke("What is artificial intelligence?")
        print("\n✅ LangChain test successful!")
        print(f"Model response: {response}")
        return True
    except Exception as e:
        print(f"\n❌ LangChain test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing token loading and usage...\n")
    
    # Test token loading
    token = test_token_loading()
    
    if token:
        # Test HuggingFace API
        if test_huggingface_api(token):
            # Test LangChain
            test_langchain(token) 