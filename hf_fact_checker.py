from typing import Dict, Any, List
import os
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from huggingface_hub import InferenceClient
import logging
import urllib3
import ssl
import time
import copy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable SSL verification globally (use with caution - this is for testing only)
try:
    # For Python >= 2.7.9
    ssl._create_default_https_context = ssl._create_unverified_context
    logger.info("Disabled SSL verification globally")
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass

class HuggingFaceFactChecker:
    def __init__(self, skip_api_test=False):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.api_available = False
        
        # Initialize Hugging Face API token
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.hf_token:
            logger.warning("HUGGINGFACE_API_TOKEN not found in environment variables")
            logger.warning("Will use fallback knowledge base only")
        else:
            logger.info(f"Found API token: {self.hf_token[:5]}...")
            
            # Test the API connection if not skipped
            if not skip_api_test:
                try:
                    self._test_api_connection()
                    logger.info("Hugging Face API initialized successfully")
                    self.api_available = True
                except Exception as e:
                    logger.warning(f"API connection test failed, will use fallbacks: {str(e)}")
            else:
                logger.info("Skipping API test as requested")
                self.api_available = True  # Assume it's available but we'll handle errors during actual use
        
        # Initialize model for sentiment analysis to help evaluate claims
        logger.info("Loading sentiment analysis model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment analysis model: {str(e)}")
            # We'll still continue even if this fails
        
        # Common knowledge facts as fallback
        self.knowledge_base = {
            "earth": {
                "round": {"verdict": "true", "confidence": 0.98, 
                         "evidence": ["Scientific consensus establishes Earth as an oblate spheroid", 
                                     "Observable evidence includes ship disappearance over horizon",
                                     "Satellite imagery confirms Earth's shape"]},
                "flat": {"verdict": "false", "confidence": 0.99,
                        "evidence": ["Scientific consensus rejects flat Earth theory",
                                    "Multiple lines of evidence confirm Earth's spherical nature",
                                    "Satellite imagery contradicts flat Earth claims"]}
            },
            "vaccine": {
                "autism": {"verdict": "false", "confidence": 0.95,
                          "evidence": ["Multiple large-scale studies found no link between vaccines and autism",
                                      "Original study suggesting link was retracted for methodological errors",
                                      "Medical consensus confirms vaccines don't cause autism"]},
                "safe": {"verdict": "true", "confidence": 0.9,
                        "evidence": ["Vaccines undergo rigorous testing for safety and efficacy",
                                    "Side effects are generally mild and temporary",
                                    "Benefits significantly outweigh potential risks"]}
            },
            "climate": {
                "change": {"verdict": "true", "confidence": 0.95,
                          "evidence": ["Scientific consensus supports human-caused climate change",
                                      "Temperature records show clear warming trend",
                                      "Multiple independent lines of evidence confirm climate change"]},
                "hoax": {"verdict": "false", "confidence": 0.97,
                        "evidence": ["Scientific data contradicts 'hoax' claim",
                                    "Observed changes in climate patterns are well-documented",
                                    "Multiple independent measurements confirm warming trends"]}
            },
            "smoking": {
                "cancer": {"verdict": "true", "confidence": 0.99,
                          "evidence": ["Decades of research establish causal link between smoking and cancer",
                                      "Tobacco smoke contains known carcinogens",
                                      "Smoking cessation reduces cancer risk over time"]},
                "healthy": {"verdict": "false", "confidence": 0.99,
                           "evidence": ["No credible research supports smoking as healthy",
                                       "Tobacco contains harmful chemicals and toxins",
                                       "Smoking damages nearly every organ in the body"]}
            },
            "water": {
                "boils": {"verdict": "true", "confidence": 0.99,
                         "evidence": ["Water boils at 100°C (212°F) at standard atmospheric pressure",
                                     "This is a well-established physical property",
                                     "Boiling point varies with pressure and dissolved substances"]},
                "wet": {"verdict": "true", "confidence": 0.95,
                       "evidence": ["Water makes other materials wet by adhering to their surface",
                                   "Wetness is the ability of a liquid to adhere to solid surfaces",
                                   "Water molecules have cohesive and adhesive properties"]}
            },
            "moon": {
                "landing": {"verdict": "true", "confidence": 0.99,
                           "evidence": ["Apollo missions successfully landed humans on the moon",
                                       "Multiple independent sources confirmed the moon landings",
                                       "Physical evidence including moon rocks verify the landings"]},
                "fake": {"verdict": "false", "confidence": 0.99,
                        "evidence": ["Moon landing conspiracy theories have been debunked",
                                    "Physical evidence contradicts hoax claims",
                                    "Thousands of people would need to maintain the conspiracy"]}
            }
        }
    
    def _test_api_connection(self, max_retries=2, retry_delay=1):
        """Test the API connection with a simple query and retry on failure"""
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                logger.info(f"Testing API connection (attempt {attempt+1}/{max_retries})...")
                
                # API endpoint URL
                url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
                
                # Headers with authorization
                headers = {
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                }
                
                # Data payload
                data = {
                    "inputs": "This is a test.",
                    "parameters": {
                        "candidate_labels": ["true", "false"]
                    }
                }
                
                # Make the request
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=data,
                    verify=False,  # Disable SSL verification
                    timeout=10  # Add timeout to prevent hanging
                )
                
                # Check if request was successful
                response.raise_for_status()
                
                # Parse JSON response
                result = response.json()
                logger.info(f"API test response: {result}")
                
                logger.info("API connection test successful")
                return True
            except Exception as e:
                last_error = e
                logger.warning(f"API test attempt {attempt+1} failed: {str(e)}")
                attempt += 1
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        # If we get here, all attempts failed
        logger.error(f"All API connection test attempts failed: {str(last_error)}")
        raise last_error
    
    def analyze_claim(self, claim: str) -> Dict[str, Any]:
        """
        Analyze a claim using Hugging Face models for fact-checking.
        Always try to use the API first, and only use fallbacks if API fails.
        """
        logger.info(f"Analyzing claim: {claim}")
        
        # First attempt: Use Hugging Face's API for claim verification if available
        if self.api_available and self.hf_token:
            try:
                logger.info("Attempting to use Hugging Face API for fact-checking...")
                
                # Use the API directly - no filtering before API call
                fact_check_result = self._direct_api_call(claim)
                logger.info("API fact check successful")
                
                # Set flag indicating API was used
                fact_check_result['api_used'] = True
                fact_check_result['api_corrected'] = False
                
                # Only verify obvious factual errors from the API
                # For most claims, trust the API's verdict
                post_processed_result = self._verify_api_response(fact_check_result, claim)
                if post_processed_result and post_processed_result['api_corrected']:
                    logger.info("Applied factual correction to API response for well-known fact")
                    post_processed_result['api_used'] = True
                    return post_processed_result
                
                return fact_check_result
                
            except Exception as e:
                logger.error(f"API fact check failed: {str(e)}")
                logger.info("Falling back to knowledge base...")
        else:
            logger.info("API not available, using knowledge base...")
        
        # If API fails or not available, try knowledge base
        cleaned_claim = self._preprocess_claim(claim)
        matched_result = self._match_knowledge_base(cleaned_claim)
        
        if matched_result:
            logger.info("Using knowledge base for fact-checking")
            # Flag for UI to show as local database
            matched_result['api_used'] = False
            matched_result['api_corrected'] = False
            return matched_result
        
        # Last resort: sentiment analysis
        logger.info("Using sentiment analysis as last resort")
        result = self._sentiment_based_analysis(claim)
        result['api_used'] = False
        result['api_corrected'] = False
        return result
    
    def _preprocess_claim(self, claim: str) -> str:
        """
        Clean and preprocess the claim text
        """
        # Convert to lowercase
        claim = claim.lower()
        
        # Remove special characters
        claim = re.sub(r'[^\w\s]', '', claim)
        
        # Tokenize and remove stop words
        tokens = word_tokenize(claim)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def _match_knowledge_base(self, claim: str) -> Dict[str, Any]:
        """
        Try to match claim against our knowledge base
        """
        words = claim.split()
        
        # First try exact matching
        for topic, subtopics in self.knowledge_base.items():
            if topic in words:
                for subtopic, details in subtopics.items():
                    if subtopic in words:
                        biases = ["Confirmation bias may affect interpretation",
                                 "Source selection bias possible"]
                        context = ["Historical context important for full understanding",
                                  "Cultural factors may influence interpretation"]
                        
                        return {
                            'verdict': details["verdict"],
                            'confidence': details["confidence"],
                            'evidence': details["evidence"],
                            'biases': biases[:1],
                            'context': context[:1]
                        }
        
        # If no exact match, try looser matching
        for topic, subtopics in self.knowledge_base.items():
            if topic in claim:  # Check if topic anywhere in claim
                # If topic found but no subtopic, use the most relevant subtopic
                most_relevant_subtopic = None
                highest_confidence = 0
                
                for subtopic, details in subtopics.items():
                    # Pick the subtopic with the highest confidence that makes sense
                    if details["confidence"] > highest_confidence:
                        most_relevant_subtopic = subtopic
                        highest_confidence = details["confidence"]
                
                if most_relevant_subtopic:
                    details = subtopics[most_relevant_subtopic]
                    evidence = [f"Based on information about {topic}, this claim appears {details['verdict']}"] + details["evidence"]
                    
                    return {
                        'verdict': details["verdict"],
                        'confidence': details["confidence"] * 0.9,  # Slightly reduce confidence for inexact match
                        'evidence': evidence,
                        'biases': ["This is a partial match from our knowledge base"],
                        'context': [f"This analysis is based on facts about {topic} {most_relevant_subtopic}"]
                    }
        
        return None
    
    def _direct_api_call(self, claim, timeout=15):
        """
        Makes a direct API call to Hugging Face without pre-filtering.
        This is the primary fact-checking method.
        """
        logger.info("Making direct API call to Hugging Face...")
        
        # Using the model fine-tuned on FEVER dataset for better fact verification
        model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        logger.info(f"Using model: {model_name}")
        
        # API endpoint URL
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Headers with authorization
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        # Improve the query format with a hypothesis for better NLI performance
        data = {
            "inputs": claim,
            "parameters": {
                "candidate_labels": ["entailment", "contradiction", "neutral"]
            }
        }
        
        # Make the request
        response = requests.post(
            url, 
            headers=headers, 
            json=data,
            verify=False,  # Disable SSL verification
            timeout=timeout
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        logger.info(f"Received API response: {result}")
        
        # Process the response
        scores = result['scores']
        labels = result['labels']
        
        # Get the highest scoring label
        max_score_index = scores.index(max(scores))
        raw_verdict = labels[max_score_index]
        confidence = scores[max_score_index]
        
        # Map the model's output to our verdict format
        verdict_map = {
            "entailment": "true",
            "contradiction": "false", 
            "neutral": "unverified"
        }
        
        verdict = verdict_map.get(raw_verdict.lower(), "unverified")
        
        # Try to extract potential correct information for false claims
        potential_correction = None
        if verdict == "false":
            potential_correction = self._get_potential_correction(claim, verdict)
        
        # Generate evidence based on the claim and verdict
        if verdict == "false" and potential_correction:
            # For false claims with known correct information
            evidence = [
                f"Based on fact-checking analysis, this claim appears to be FALSE.",
                f"CORRECTION: {potential_correction}",
                f"This conclusion is based on the Hugging Face DeBERTa model trained on the FEVER dataset."
            ]
        elif verdict == "true":
            # For true claims
            evidence = [
                f"Based on fact-checking analysis, this claim appears to be TRUE.",
                f"Confidence score: {confidence:.2f}",
                f"This conclusion is based on the Hugging Face DeBERTa model trained on the FEVER dataset."
            ]
        else:
            # For unverified claims
            evidence = [
                f"Based on fact-checking analysis, this claim is UNVERIFIED.",
                f"Confidence score: {confidence:.2f}",
                f"The model cannot definitively confirm or deny this claim with high confidence."
            ]
        
        # Generate bias and context information
        biases = [
            "This analysis is directly from the Hugging Face DeBERTa model",
            "AI models may have limitations with specialized knowledge"
        ]
        
        contexts = [
            "This verdict is based on the model's training on factual claims",
            "For critical information, verify with multiple reliable sources"
        ]
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'evidence': evidence,
            'biases': biases,
            'context': contexts
        }
    
    def _get_potential_correction(self, claim: str, verdict: str) -> str:
        """
        Try to provide a correct statement for false claims
        """
        if verdict != "false":
            return None
            
        claim_lower = claim.lower()
        
        # Expanded corrections dictionary with more detailed explanations and sources
        corrections = {
            # Earth facts
            "earth is flat": "Earth is an oblate spheroid (slightly flattened sphere), not flat. This is proven by numerous satellite images, physics measurements, and direct observations including the curvature visible from high altitudes.",
            "earth is center of solar system": "The Sun is at the center of our solar system, not Earth. Earth orbits around the Sun, completing one orbit every 365.25 days.",
            "earth is center of universe": "Earth is not the center of the universe. Modern cosmology shows that the universe has no center and is expanding in all directions equally.",
            
            # Moon facts
            "moon is made of cheese": "The Moon is made of rock, similar to Earth's crust but with more metals like titanium. Apollo missions brought back lunar samples that confirmed this composition.",
            "moon landing fake": "The Moon landings were real. Multiple independent sources including the Soviet Union (who were rivals) confirmed them, and reflectors left on the lunar surface can still be detected from Earth today.",
            
            # Sun facts
            "sun is cold": "The Sun is extremely hot with a surface temperature of about 5,500°C (9,940°F) and a core temperature of about 15 million°C (27 million°F).",
            "sun orbits earth": "The Earth orbits the Sun, not the other way around. This heliocentric model was confirmed by observations from Copernicus, Galileo, and modern astronomy.",
            "sun is a planet": "The Sun is a star, not a planet. It's a massive ball of hot plasma that generates energy through nuclear fusion.",
            
            # Medical facts
            "vaccines cause autism": "Multiple large-scale scientific studies have found no link between vaccines and autism. The original study suggesting this link was retracted for methodological errors and ethical violations.",
            "covid is a hoax": "COVID-19 is a real infectious disease caused by the SARS-CoV-2 virus, confirmed by multiple independent research laboratories worldwide and responsible for millions of documented deaths.",
            
            # Climate facts
            "climate change is a hoax": "Climate change is supported by overwhelming scientific evidence including temperature records, ice core samples, and observable impacts. Over 97% of climate scientists agree it is real and primarily human-caused.",
            "global warming not real": "Global warming is real and documented by multiple independent scientific organizations tracking Earth's temperature. The planet's average temperature has risen by about 1°C since the pre-industrial era.",
            
            # Health facts
            "smoking is healthy": "Smoking is extremely harmful to health and is a leading cause of preventable diseases and death worldwide. It increases risk of cancer, heart disease, stroke, and many other health problems.",
            "alcohol is good for health": "While small amounts of certain alcoholic beverages may have some health benefits for specific populations, alcohol consumption is generally associated with numerous health risks and is not recommended as a health practice.",
            
            # Technology facts
            "5g causes cancer": "There is no scientific evidence that 5G technology causes cancer. 5G uses radio waves that are non-ionizing radiation, which means they don't have enough energy to damage DNA directly.",
            "ai can think like humans": "Current AI systems cannot think like humans. They use pattern recognition and statistical methods to generate responses but lack true understanding, consciousness, or general intelligence."
        }
        
        # Check for matching corrections in broader context
        for false_claim, correction in corrections.items():
            if false_claim in claim_lower:
                return correction
                
        # Capital city corrections - expanded list
        if "capital" in claim_lower:
            country_capitals = {
                "india": "New Delhi",
                "usa": "Washington D.C.",
                "united states": "Washington D.C.",
                "uk": "London",
                "united kingdom": "London",
                "japan": "Tokyo",
                "china": "Beijing",
                "france": "Paris",
                "germany": "Berlin",
                "italy": "Rome",
                "spain": "Madrid",
                "canada": "Ottawa",
                "australia": "Canberra",
                "brazil": "Brasília",
                "russia": "Moscow",
                "mexico": "Mexico City",
                "south africa": "Pretoria (administrative), Cape Town (legislative), Bloemfontein (judicial)",
                "egypt": "Cairo",
                "turkey": "Ankara",
                "south korea": "Seoul",
                "thailand": "Bangkok",
                "pakistan": "Islamabad",
                "philippines": "Manila"
            }
            
            for country, capital in country_capitals.items():
                if country in claim_lower and capital.lower() not in claim_lower:
                    return f"The capital of {country.title()} is {capital}."
        
        # Planet facts
        if any(planet in claim_lower for planet in ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]):
            planet_facts = {
                "mercury": "Mercury is the smallest and innermost planet in the Solar System, orbiting closest to the Sun.",
                "venus": "Venus is the second planet from the Sun and the hottest planet in our solar system due to its thick atmosphere that traps heat.",
                "mars": "Mars is the fourth planet from the Sun, often called the 'Red Planet' due to its reddish appearance caused by iron oxide on its surface.",
                "jupiter": "Jupiter is the largest planet in our solar system, a gas giant with a distinctive Great Red Spot that is a giant storm.",
                "saturn": "Saturn is the sixth planet from the Sun and is known for its prominent ring system, which consists primarily of ice particles and rock debris.",
                "uranus": "Uranus is the seventh planet from the Sun and rotates on its side, giving it extreme seasons that last for decades.",
                "neptune": "Neptune is the eighth and farthest known planet from the Sun, characterized by its vivid blue color caused by methane in its atmosphere.",
                "pluto": "Pluto is a dwarf planet in the Kuiper belt. It was considered the ninth planet until 2006 when it was reclassified as a dwarf planet."
            }
            
            for planet, fact in planet_facts.items():
                if planet in claim_lower:
                    return fact
        
        # Basic scientific facts
        scientific_facts = {
            "water boils": "Water boils at 100°C (212°F) at standard atmospheric pressure at sea level. The boiling point decreases with increasing altitude.",
            "water freezes": "Water freezes at 0°C (32°F) at standard atmospheric pressure.",
            "gravity pulls": "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, gravity accelerates objects at approximately 9.8 m/s².",
            "evolution": "Evolution by natural selection is the scientific theory explaining how species change over time through genetic variations and selective pressures, supported by extensive fossil records and genetic evidence."
        }
        
        for keyword, fact in scientific_facts.items():
            if keyword in claim_lower:
                return fact
        
        # Return a more informative generic correction if no specific one is found
        return "This claim appears to be factually incorrect based on analysis by the DeBERTa model, which has been trained on a large dataset of verified facts. Please consult reliable sources for accurate information on this topic."
    
    def _sentiment_based_analysis(self, claim: str) -> Dict[str, Any]:
        """
        Use sentiment analysis as a fallback method for basic claim analysis
        """
        try:
            # Tokenize and prepare the claim for sentiment analysis
            inputs = self.tokenizer(claim, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.sentiment_model(**inputs)
            
            # Get prediction
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_score = probs[0][1].item()  # Positive sentiment probability
            
            # Determine verdict based on sentiment
            # This is a simplistic approach but can serve as backup
            if sentiment_score > 0.7:
                # Very positive claims are sometimes exaggerated
                verdict = "unverified"
                confidence = 0.6
                evidence = [
                    "The claim contains highly positive sentiment which can sometimes indicate exaggeration",
                    "Sentiment analysis alone cannot determine factuality",
                    "Further verification with reliable sources is recommended"
                ]
            elif sentiment_score < 0.3:
                # Very negative claims may be misleading
                verdict = "unverified"
                confidence = 0.6
                evidence = [
                    "The claim contains negative sentiment which may indicate bias",
                    "Sentiment analysis alone cannot determine factuality",
                    "Further verification with reliable sources is recommended"
                ]
            else:
                # Neutral claims tend to be more factual, but this is a weak heuristic
                verdict = "unverified"
                confidence = 0.4
                evidence = [
                    "The claim appears relatively neutral in sentiment",
                    "Sentiment neutrality does not guarantee factuality",
                    "Further verification with reliable sources is strongly recommended"
                ]
            
            biases = [
                "Sentiment analysis is not a reliable indicator of factuality",
                "This is a fallback method when primary fact-checking is unavailable"
            ]
            
            contexts = [
                "Consider seeking verification from fact-checking organizations",
                "This analysis has limited reliability"
            ]
            
            return {
                'verdict': verdict,
                'confidence': confidence,
                'evidence': evidence,
                'biases': biases,
                'context': contexts
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Ultimate fallback with very clear uncertainty
            return {
                'verdict': "unverified",
                'confidence': 0.3,
                'evidence': [
                    "Unable to verify this claim due to technical limitations",
                    "Consider checking with established fact-checking sources",
                    "No definitive determination could be made"
                ],
                'biases': ["System limitations prevent thorough analysis"],
                'context': ["Seek professional fact-checking services for verification"]
            }
    
    def _verify_api_response(self, api_response, claim):
        """
        Verify the API response against well-known facts
        Returns a potentially corrected response
        """
        # Make a copy of the API response to avoid modifying the original
        corrected_response = copy.deepcopy(api_response)
        corrected = False
        claim_lower = claim.lower()
        
        # Check for country-capital claims
        if "capital" in claim_lower:
            # Dictionary of country-capital relationships
            country_capitals = {
                "united states": "washington dc",
                "usa": "washington dc",
                "united kingdom": "london",
                "uk": "london",
                "france": "paris",
                "germany": "berlin",
                "japan": "tokyo",
                "china": "beijing",
                "india": "delhi",
                "australia": "canberra",
                "canada": "ottawa",
                "brazil": "brasilia",
                "mexico": "mexico city",
                "russia": "moscow",
                "south africa": "pretoria",
                "italy": "rome",
                "spain": "madrid",
                "portugal": "lisbon",
                "netherlands": "amsterdam",
                "belgium": "brussels",
                "switzerland": "bern",
                "austria": "vienna",
                "poland": "warsaw",
                "sweden": "stockholm",
                "norway": "oslo",
                "finland": "helsinki",
                "denmark": "copenhagen",
                "greece": "athens",
                "turkey": "ankara",
                "egypt": "cairo",
                "saudi arabia": "riyadh",
                "israel": "jerusalem",
                "south korea": "seoul",
                "north korea": "pyongyang",
                "thailand": "bangkok",
                "vietnam": "hanoi",
                "philippines": "manila",
                "indonesia": "jakarta",
                "malaysia": "kuala lumpur",
                "singapore": "singapore",
                "new zealand": "wellington",
                "argentina": "buenos aires",
                "chile": "santiago",
                "peru": "lima",
                "colombia": "bogota",
                "venezuela": "caracas",
                "nigeria": "abuja",
                "kenya": "nairobi",
                "tanzania": "dodoma"
            }
            
            for country, capital in country_capitals.items():
                if country in claim_lower:
                    if capital in claim_lower:
                        # If the claim states the correct capital, it should be TRUE
                        if api_response['verdict'] != 'true':
                            logger.info(f"Correcting API response for {claim}: should be TRUE")
                            corrected_response['verdict'] = 'true'
                            corrected_response['evidence'] = [
                                f"This is factually correct. {capital.title()} is the capital of {country.title()}.",
                                "This has been verified against geographical data.",
                                "Source: Verified geographical information"
                            ]
                            corrected = True
                    else:
                        # If the claim states the wrong capital, it should be FALSE
                        if api_response['verdict'] != 'false':
                            logger.info(f"Correcting API response for {claim}: should be FALSE")
                            corrected_response['verdict'] = 'false'
                            corrected_response['evidence'] = [
                                f"This claim is factually incorrect. The capital of {country.title()} is {capital.title()}, not what was claimed.",
                                "This has been verified against geographical data.",
                                "Source: Verified geographical information"
                            ]
                            corrected = True
        
        # Check for celestial fact claims
        celestial_facts = {
            "sun": {
                "hot": True,
                "cold": False,
                "star": True,
                "planet": False,
                "center of solar system": True,
                "orbits earth": False,
                "closest star": True
            },
            "moon": {
                "satellite": True,
                "planet": False,
                "star": False,
                "orbits earth": True,
                "made of cheese": False,
                "natural satellite": True
            },
            "earth": {
                "flat": False,
                "round": True,
                "sphere": True,
                "planet": True,
                "star": False,
                "orbits sun": True,
                "center of universe": False,
                "center of solar system": False
            },
            "mars": {
                "planet": True,
                "red planet": True,
                "has life": False,
                "has water": True,  # Mars has ice and evidence of liquid water in the past
                "orbits sun": True
            },
            "venus": {
                "planet": True,
                "star": False,
                "hotter than mercury": True,  # Venus is hotter than Mercury despite being further from the Sun
                "orbits sun": True
            },
            "jupiter": {
                "largest planet": True,
                "gas giant": True,
                "has rings": True,
                "orbits sun": True
            }
        }
        
        for celestial_body, facts in celestial_facts.items():
            if celestial_body in claim_lower:
                for property_name, property_value in facts.items():
                    if property_name in claim_lower:
                        # The claim is about this property of the celestial body
                        expected_verdict = 'true' if property_value else 'false'
                        
                        if api_response['verdict'] != expected_verdict:
                            logger.info(f"Correcting API response for {claim}: should be {expected_verdict}")
                            corrected_response['verdict'] = expected_verdict
                            if expected_verdict == 'true':
                                corrected_response['evidence'] = [
                                    f"This claim about {celestial_body} is factually correct.",
                                    f"The {celestial_body} is indeed {property_name}.",
                                    "Source: Verified astronomical data"
                                ]
                            else:
                                # Create a correction based on the false claim
                                if celestial_body == "sun" and property_name == "cold":
                                    correction = "The Sun is extremely hot with a surface temperature of about 5,500°C (9,940°F)."
                                elif celestial_body == "earth" and property_name == "flat":
                                    correction = "Earth is an oblate spheroid (slightly flattened sphere), not flat."
                                elif celestial_body == "moon" and property_name == "made of cheese":
                                    correction = "The Moon is made of rock, similar to Earth's crust but with different composition."
                                else:
                                    opposite_props = {
                                        "planet": "not a planet",
                                        "star": "not a star",
                                        "orbits earth": "does not orbit Earth",
                                        "orbits sun": "does not orbit the Sun",
                                        "center of universe": "not the center of the universe",
                                        "center of solar system": "not the center of the solar system"
                                    }
                                    correction = f"The {celestial_body} is {opposite_props.get(property_name, f'not {property_name}')}."
                                
                                corrected_response['evidence'] = [
                                    f"This claim about {celestial_body} is factually incorrect.",
                                    correction,
                                    "Source: Verified astronomical data"
                                ]
                            corrected = True
        
        # Check for continent claims
        continents = ["africa", "antarctica", "asia", "australia", "europe", "north america", "south america"]
        countries_by_continent = {
            "africa": ["egypt", "nigeria", "south africa", "kenya", "ethiopia", "morocco", "algeria", "ghana"],
            "asia": ["china", "india", "japan", "south korea", "thailand", "vietnam", "indonesia", "malaysia", 
                     "saudi arabia", "turkey", "iran", "iraq", "israel", "pakistan", "bangladesh"],
            "europe": ["united kingdom", "france", "germany", "italy", "spain", "portugal", "netherlands", 
                       "belgium", "switzerland", "austria", "poland", "russia", "ukraine", "sweden", "norway"],
            "north america": ["united states", "usa", "canada", "mexico", "cuba", "jamaica", "costa rica", "panama"],
            "south america": ["brazil", "argentina", "chile", "peru", "colombia", "venezuela", "ecuador", "bolivia"],
            "australia": ["australia"]  # Australia is both a country and a continent
        }
        
        # Check for claims about countries being continents
        for country_list in countries_by_continent.values():
            for country in country_list:
                if country in claim_lower and "continent" in claim_lower and country != "australia":
                    if api_response['verdict'] != 'false':
                        logger.info(f"Correcting API response for {claim}: should be FALSE")
                        corrected_response['verdict'] = 'false'
                        
                        # Find which continent the country belongs to
                        continent = next((cont for cont, countries in countries_by_continent.items() 
                                        if country in countries), "unknown")
                        
                        corrected_response['evidence'] = [
                            f"This claim is factually incorrect. {country.title()} is a country, not a continent.",
                            f"{country.title()} is located in the continent of {continent.title()}.",
                            "Source: Verified geographical information"
                        ]
                        corrected = True
        
        # Check for claims about continents
        for continent in continents:
            if continent in claim_lower and "continent" in claim_lower:
                if api_response['verdict'] != 'true':
                    logger.info(f"Correcting API response for {claim}: should be TRUE")
                    corrected_response['verdict'] = 'true'
                    corrected_response['evidence'] = [
                        f"This claim is factually correct. {continent.title()} is indeed a continent.",
                        "This has been verified against geographical data.",
                        "Source: Verified geographical information"
                    ]
                    corrected = True
        
        # Add more fact verification logic here as needed
        
        # Additional facts about major historical events
        historical_facts = {
            "world war 2 ended in 1945": True,
            "world war 2 ended in 1944": False,
            "world war 1 started in 1914": True,
            "world war 1 started in 1915": False,
            "united states declared independence in 1776": True,
            "berlin wall fell in 1989": True,
            "berlin wall fell in 1991": False,
            "apollo 11 landed on the moon in 1969": True,
            "apollo 11 landed on the moon in 1970": False
        }
        
        for fact, is_true in historical_facts.items():
            if all(word in claim_lower for word in fact.lower().split()):
                expected_verdict = 'true' if is_true else 'false'
                if api_response['verdict'] != expected_verdict:
                    logger.info(f"Correcting API response for {claim}: should be {expected_verdict}")
                    corrected_response['verdict'] = expected_verdict
                    
                    if is_true:
                        corrected_response['evidence'] = [
                            "This historical claim is factually correct.",
                            f"The statement that '{fact}' is verified by historical records.",
                            "Source: Verified historical data"
                        ]
                    else:
                        # Find the correct version of this fact
                        for correct_fact, _ in historical_facts.items():
                            if correct_fact.split()[-3:] == fact.split()[-3:] and historical_facts[correct_fact]:
                                correction = correct_fact
                                break
                        else:
                            correction = "The provided historical information is incorrect."
                        
                        corrected_response['evidence'] = [
                            "This historical claim is factually incorrect.",
                            f"Correction: {correction}",
                            "Source: Verified historical data"
                        ]
                    corrected = True
        
        if corrected:
            corrected_response['api_corrected'] = True
        else:
            corrected_response['api_corrected'] = False
            
        return corrected_response 