"""
Exercise 4.5: ODD Coverage with k-Projection Coverage
"""

import sys
import os

# Ensure src is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from odd_coverage.kprojection import KProjectionCoverage

def main():
    print("--- Exercise 4.5: ODD Coverage ---")
    
    # ODD Description from Exercise 2.2
    odd_description = {
        "weather": ["Clear", "Sunny", "Cloudy"],
        "lighting": ["Daytime"],
        "camera": ["Forward-facing"],
        "scene_type": ["Urban environment"],
        "vehicle_speed": ["0-25 km/h", "25-50 km/h"],
    }
    
    # The 'test/test' set is a single simulation run. It covers only one specific
    # combination of the ODD dimensions.
    # From CARLA defaults for clear daytime in a town, speeds vary but we'll assume 
    # it covers the speed range during the drive. Let's add the scenarios that represent the test set.
    # To be generous, we assume the test set covers both speed bins under clear daytime urban conditions.
    test_scenarios = [
        {"weather": "Clear", "lighting": "Daytime", "camera": "Forward-facing", "scene_type": "Urban environment", "vehicle_speed": "0-25 km/h"},
        {"weather": "Clear", "lighting": "Daytime", "camera": "Forward-facing", "scene_type": "Urban environment", "vehicle_speed": "25-50 km/h"},
    ]
    
    for k in [1, 2, 3]:
        cov = KProjectionCoverage(k=k, desc=odd_description)
        for scenario in test_scenarios:
            cov.add_scenario(scenario)
        
        result = cov.compute()
        print(f"{k}-Projection Coverage: {result.coverage:.2%}")

if __name__ == "__main__":
    main()
