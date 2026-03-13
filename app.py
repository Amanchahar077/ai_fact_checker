from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import requests
import logging
from hf_fact_checker import HuggingFaceFactChecker
from news_verifier import NewsVerifier
from source_validator import SourceValidator

try:
    from waitress import serve
except ImportError:
    serve = None

# Load environment variables from .env file
load_dotenv()
logging.getLogger("waitress").setLevel(logging.WARNING)

app = Flask(__name__)

# Initialize the fact checker
fact_checker = HuggingFaceFactChecker(skip_api_test=True)

# Create minimal instances of other components
news_verifier = NewsVerifier()
source_validator = SourceValidator()

@app.route('/')
def home():
    return render_template('index.html')

def _api_test_allowed() -> bool:
    allow_flag = os.getenv("ALLOW_API_TEST", "").strip().lower() in {"1", "true", "yes"}
    return request.remote_addr in {"127.0.0.1", "::1"} or allow_flag

@app.route('/api_test', methods=['GET'])
def api_test():
    """Diagnostic endpoint to test API connectivity"""
    if not _api_test_allowed():
        return jsonify({
            'status': 'error',
            'message': 'API test is disabled'
        }), 403

    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    try:
        if not api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN is not configured")

        test_response = requests.post(
            "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            },
            json={"inputs": "This is a test.", "parameters": {"candidate_labels": ["true", "false"]}},
            verify=fact_checker.verify_ssl,
            timeout=15
        )
        test_response.raise_for_status()
        
        # Return the response and status
        return jsonify({
            'status': 'success',
            'message': 'API connection successful'
        })
    except requests.HTTPError as e:
        message = fact_checker._explain_hf_http_error(e)
        return jsonify({
            'status': 'error',
            'message': message
        }), 500
    except Exception as e:
        # Return error information
        return jsonify({
            'status': 'error',
            'message': f'API connection failed: {str(e)}'
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
            'api_corrected': result.get('api_corrected', False),
            'api_error': result.get('api_error')
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

@app.errorhandler(404)
def not_found(_error):
    if request.path in {"/check_fact", "/api_test"}:
        return jsonify({
            "status": "error",
            "message": "Endpoint not found. Verify the Render service is running the Flask app."
        }), 404
    return jsonify({"status": "error", "message": "Not found"}), 404

@app.errorhandler(405)
def method_not_allowed(_error):
    if request.path in {"/check_fact", "/api_test"}:
        return jsonify({
            "status": "error",
            "message": "Method not allowed. Use POST for /check_fact and GET for /api_test."
        }), 405
    return jsonify({"status": "error", "message": "Method not allowed"}), 405

if __name__ == '__main__':
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    print("AI Fact Checker is running!")
    print(f"Listening on http://{host}:{port}")
    if serve:
        serve(app, host=host, port=port)
    else:
        app.run(debug=False, use_reloader=False, host=host, port=port)
