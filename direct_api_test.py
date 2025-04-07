"""
Simple, direct test for Hugging Face API connectivity
This file has minimal dependencies and is used to diagnose API connection issues
"""

import os
import requests
import json
import ssl
from dotenv import load_dotenv

# Disable SSL verification for testing
ssl._create_default_https_context = ssl._create_unverified_context

def test_direct_api():
    """Test the API using direct HTTP requests"""
    # Load token from .env
    load_dotenv()
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    
    if not api_token:
        print("❌ ERROR: No API token found in .env file")
        return False
    
    print(f"✅ Found API token: {api_token[:5]}...")
    
    # API endpoint URL
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    
    # Headers with authorization
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # Data payload
    data = {
        "inputs": "The earth is round.",
        "parameters": {
            "candidate_labels": ["true", "false", "unverified"]
        }
    }
    
    try:
        print("🔄 Sending request to Hugging Face API...")
        response = requests.post(
            url, 
            headers=headers, 
            json=data,
            verify=False,  # Disable SSL verification
            timeout=15
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        print("✅ API request successful!")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {json.dumps(result, indent=2)}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return False

if __name__ == "__main__":
    print("\n=== DIRECT HUGGING FACE API TEST ===\n")
    success = test_direct_api()
    
    if success:
        print("\n✅ API TEST PASSED: Connection successful")
    else:
        print("\n❌ API TEST FAILED: Could not connect to Hugging Face API")
        print("\nTroubleshooting steps:")
        print("1. Check if your API token is valid at huggingface.co/settings/tokens")
        print("2. Check your internet connection")
        print("3. Check if Hugging Face API servers are up") 