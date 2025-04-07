import random
from datetime import datetime, timedelta

class NewsVerifier:
    def __init__(self):
        # Simulated news sources
        self.news_sources = [
            {"name": "The Fact Journal", "reliability": 0.92},
            {"name": "Truth Daily", "reliability": 0.88},
            {"name": "News Analyzer", "reliability": 0.85},
            {"name": "Science Today", "reliability": 0.95},
            {"name": "Global Report", "reliability": 0.82}
        ]
    
    def verify_claim(self, claim):
        """
        Verify a claim against news sources.
        In this simplified version, we return mock data.
        """
        # Generate a reliability score based on claim characteristics
        # Just for simulation purposes
        if any(keyword in claim.lower() for keyword in ["proven", "scientific", "study", "research"]):
            reliability = random.uniform(0.65, 0.9)
        elif any(keyword in claim.lower() for keyword in ["conspiracy", "secret", "they don't want you to know"]):
            reliability = random.uniform(0.2, 0.4)
        else:
            reliability = random.uniform(0.4, 0.7)
            
        # Generate random recent dates for articles
        current_date = datetime.now()
        dates = [(current_date - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d") for _ in range(3)]
        
        # Generate mock sources based on the claim
        words = claim.split()
        relevant_words = [word for word in words if len(word) > 3][:3]
        
        sources = [
            {
                "title": f"Analysis of '{' '.join(relevant_words)}' claim reveals important insights",
                "source": self.news_sources[0]["name"],
                "date": dates[0],
                "url": "https://example.com/analysis"
            },
            {
                "title": f"Fact checking: '{' '.join(relevant_words)}'",
                "source": self.news_sources[1]["name"],
                "date": dates[1],
                "url": "https://example.com/factcheck"
            },
            {
                "title": f"Expert opinions on '{' '.join(relevant_words)}'",
                "source": self.news_sources[2]["name"], 
                "date": dates[2],
                "url": "https://example.com/expert-opinion"
            }
        ]
        
        return {
            "reliability": reliability,
            "summary": f"Based on analysis of multiple news sources, this claim has a reliability score of {reliability:.2f}. " +
                      f"Multiple sources have discussed this topic with varying degrees of support for the claim.",
            "sources": sources
        } 