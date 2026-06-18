# Quantitative Projection Coverage 

Implementation of Quantitative Projection Coverage from the paper "Quantitative Projection Coverage for Testing ML-enabled Autonomous Systems" by Chih-Hong Cheng _et al._ available at [arxiv](https://arxiv.org/abs/1805.04333).

## Usage 

```python
from kprojection import KProjectionCoverage

# Create an Operational Design Domain (ODD) description
# This is a dict with one entry per dimension, which maps each dimension to possible values 
description = {
    "weather": ["good", "bad", "ugly"],
    "temperature": [1, 2, 3, 4],
    "humidity": [0.1, 0.2, 0.3, 0.4, 0.5],
}

# Create the metric for some value of k-way coverage 
cov = KProjectionCoverage(k=2, desc=description)

# Add covered scenarios to the metric
# Each scenario assigns a value to each dimension in the ODD description 
cov.add_scenario({"weather": "bad", "temperature": 1, "humidity": 0.1})
cov.add_scenario({"weather": "ugly", "temperature": 2, "humidity": 0.1})
cov.add_scenario({"weather": "good", "temperature": 2, "humidity": 0.1})
cov.add_scenario({"weather": "good", "temperature": 3, "humidity": 0.1})
cov.add_scenario({"weather": "good", "temperature": 4, "humidity": 0.1})

# Compute the results 
print(cov.compute())
```

## Features 
The implementation currently lacks the following features:
- Defining a specific number of hits required before a particular setting is considered "covered". Currently, 1 hit is sufficient for coverage. 
- Resticting the number of coverable setting
