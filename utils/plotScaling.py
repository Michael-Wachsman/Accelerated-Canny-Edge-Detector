import matplotlib.pyplot as plt
import numpy as np
import os, sys

def read_data(filename):
    num_n_arr = []
    exec_times = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # if float(parts[1]) < 1.0 or float(parts[1]) > 3.0:
            #     continue
            num_n_arr.append(float(parts[0]))
            exec_times.append(float(parts[1])) 
    return num_n_arr, exec_times

plt.figure(figsize=(8, 5))
plot_type = sys.argv[1]
input_file = sys.argv[2]
num_threads, output_arr = read_data(input_file)

plot_name = ""
ideal_arr = []
if plot_type == 'strong':
    for n in num_threads:
        ideal_arr.append(output_arr[0] / n)
    plot_name = 'plots/strongScaling.png'
else:
    for n in num_threads:
        ideal_arr.append(output_arr[0])
    plot_name = 'plots/weakScaling.png'

plt.plot(num_threads, output_arr, marker='o', linestyle='-', color='b', label='Actual Time')
plt.plot(num_threads, ideal_arr, marker='s', linestyle='dotted', color='green', label='Ideal Time')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')
plt.title('Threads vs Execution Time')
plt.legend()
plt.grid(True)
plt.savefig(plot_name)
print(f"Plot saved as {plot_name}")