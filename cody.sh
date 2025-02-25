#!/bin/bash

# echo "-------Generating Images-------"
# python3 ./utils/imageGenerate.py weak

echo "-------Measuring Runtime-------"
python3 utils/time.py separable_parallel 

echo "-------Plotting Scaling-------"
python3 utils/plotNpxVSRuntime.py 