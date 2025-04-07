from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from hf_fact_checker import HuggingFaceFactChecker
from news_verifier import NewsVerifier
from source_validator import SourceValidator

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize the fact checker
fact_checker = HuggingFaceFactChecker(skip_api_test=True)

# Create minimal instances of other components
news_verifier = NewsVerifier()
source_validator = SourceValidator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api_test', methods=['GET'])
def api_test():
    """Diagnostic endpoint to test API connectivity"""
    try:
        # Get the API token
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        # Create a temporary direct client for testing
        from huggingface_hub import InferenceClient
        test_client = InferenceClient(token=api_token)
        
        # Test with a simple query
        test_response = test_client.post(
            model="facebook/bart-large-mnli",
            data={"inputs": "This is a test.", "parameters": {"candidate_labels": ["true", "false"]}},
            timeout=15
        )
        
        # Return the response and status
        return jsonify({
            'status': 'success',
            'message': 'API connection successful',
            'token_starts_with': api_token[:5] + '...' if api_token else 'No token',
            'token_length': len(api_token) if api_token else 0,
            'response': test_response
        })
    except Exception as e:
        # Return detailed error information
        return jsonify({
            'status': 'error',
            'message': f'API connection failed: {str(e)}',
            'token_starts_with': api_token[:5] + '...' if api_token else 'No token',
            'token_length': len(api_token) if api_token else 0,
            'error_type': type(e).__name__,
            'error_details': str(e)
        }), 500

@app.route('/check_fact', methods=['POST'])
def check_fact():
    # Get the claim from the request
    data = request.get_json()
    claim = data.get('claim', '')
    
    # Log the claim for debugging
    print(f"Received claim for fact checking: {claim}")
    
    try:
        # Process the claim with our fact checker
        result = fact_checker.analyze_claim(claim)
        
        # Make sure we return the data in the format expected by the frontend
        response = {
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'evidence': result['evidence'],
            'api_used': result.get('api_used', False),
            'api_corrected': result.get('api_corrected', False)
        }
        
        # Log the response for debugging
        print(f"Sending response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        # Log the error and return a friendly error message
        print(f"Error processing claim: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your claim.',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("AI Fact Checker is running!")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True) 