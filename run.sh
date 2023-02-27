#!/bin/bash

# Get all dense network results
cd dense_network
python dense.py

# Get all otuput-split network results
cd ../split_network
python split.py

# Get shallow network results
cd ../shallow_network
python shallow.py

# Plot other learning rules dynamics
cd ..
cd other_rules/anti_hebbian
python dense_hebb.py
cd ../contrastive_hebbian
python dense_hebb.py
cd ../hebbian
python dense_hebb.py
cd ../predictive_coding
python dense_hebb.py
cd ../..

# Print full-rank probabilities
python sample_rows.py 3
python sample_rows.py 4
