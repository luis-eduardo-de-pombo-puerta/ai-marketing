from dotenv import load_dotenv
import os
from huggingface_hub import HfApi

# Load environment variables from .env file
load_dotenv()

# Get the token
token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if token:
    print("✅ Token loaded successfully!")
    print(f"Token length: {len(token)} characters")
    
    # Test the token by making a simple API call
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        print("\n✅ Token is valid!")
        print(f"Username: {user_info['name']}")
        print(f"Email: {user_info['email']}")
    except Exception as e:
        print("\n❌ Error testing token:")
        print(f"Error: {str(e)}")
else:
    print("❌ Token not found in environment variables!") 