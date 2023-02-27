#!/bin/bash
# Get the CMNIST Results
cd cmnist
cd dense
python cmnist_dense.py
cd ../split
python cmnist_split.py
cd ..
python replot.py
cd ..
