import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

filenames = ['./time/serial.txt', './time/block_serial.txt', './time/separable_serial.txt', './time/separable_serial_CF.txt', './time/separable_parallel.txt', './time/h_gpu.txt', './time/h_gpu_sep.txt']
linenames = ['Basic_Serial', 'Block_Serial', 'Seperable_Serial', 'Seperable_Serial_wCF', 'Separable_Parallel', 'Basic_GPU', 'Separable_GPU']
COMP = 5
plot_name = './plots/linear_nvt_'+str(COMP)+'_comparison.png'
colors = ['red', 'blue', 'green', 'c', 'm', 'y', 'black']

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

for i, file in enumerate(filenames[:COMP]):
    if i <= 2: continue
    num_part, exec_times = read_data(file)
    plt.plot(num_part, exec_times, marker='o', linestyle='-', color=colors[i], label=f'{linenames[i]}')

# Plot Labels and Legend
plt.xlabel('Number of pixels')
plt.ylabel('Execution Time (s)')
plt.title('Number of pixels vs Execution Time')

plt.legend()
plt.grid(True)
plt.savefig(plot_name)
print(f"Plot saved as {plot_name}")