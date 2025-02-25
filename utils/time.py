import subprocess
import time
import numpy as np
import sys
import os
from heapq import *
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python3 time.py <build>")
    sys.exit(1)

build = sys.argv[1]
output_filename = f'time/{build}.txt'

INPUT_FOLDER = './imgs/tiled_images'

with open(output_filename, 'w') as f:
    pq = []
    # for OMP
    env = os.environ.copy()
    env['SLURM_CPU_BIND'] = 'cores'
    env['OMP_PROC_BIND'] = 'spread'
    env['OMP_PLACES'] = 'threads'
    env['OMP_NUM_THREADS'] = '32'
    for pic in os.listdir(INPUT_FOLDER):
        input_filename = os.path.join(INPUT_FOLDER, pic)
        num_pixels = None
        try:
            # Open the image file
            with Image.open(input_filename) as img:
                width, height = img.size  # Get dimensions (width x height)
                print(f"Running with picture: {pic} - Dimensions: {width} x {height}")
                num_pixels = width * height
        except Exception as e:
            print(f"Error processing {pic}: {e}")
            exit()
        
        # Only for OMP
        process = subprocess.Popen([f'./build/{build}', '--input', input_filename],
                                    stderr=subprocess.PIPE, text=True, env=env)


        # process = subprocess.Popen([f'./build/{build}', '--input', input_filename],
        #                            stderr=subprocess.PIPE, text=True)
            
        last_line = None
        
        for line in iter(process.stderr.readline, ''):
            last_line = line.strip()
        
        process.wait()

        # Add the results to min heap
        heappush(pq, (num_pixels, last_line))
        print(f"Recorded num_pixels: {num_pixels}, Output runtime: {last_line}\n")
    while pq:
        num_px, rt = heappop(pq)
        f.write(f"{num_px} {rt}\n")

print(f"Results saved to {output_filename}")
