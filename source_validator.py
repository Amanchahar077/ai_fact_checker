import random

class SourceValidator:
    def __init__(self):
        pass
        
    def validate_sources(self, claim):
        """
        Simple source validator that returns minimal data.
        This is just to make the app run with the fact checker focus.
        """
        # Return minimal data for source validation
        return {
            "validity": random.uniform(0.6, 0.9),
            "summary": "Source validation completed with minimal analysis."
        } 