import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

filenames = ['./time/serial.txt', './time/block_serial.txt', './time/separable_serial.txt', './time/separable_serial_CF.txt', './time/separable_parallel.txt', './time/h_gpu.txt', './time/gpu_sep.txt']
linenames = ['Basic_Serial', 'Block_Serial', 'Seperable_Serial', 'Seperable_Serial_wCF', 'Separable_Parallel', 'Basic_GPU', 'Separable_GPU']
COMP = 5
plot_name = './plots/loglog_'+str(COMP)+'_comparison.png'
colors = ['red', 'blue', 'green', 'c', 'm', 'y', 'black']

def read_data(filename):
    num_n_arr = []
    exec_times = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            num_n_arr.append(float(parts[0]))
            exec_times.append(float(parts[1])) 
    return num_n_arr, exec_times

plt.figure(figsize=(8, 5))

for i, file in enumerate(filenames[:COMP]):
    num_part, exec_times = read_data(file)

    log_part = np.log10(num_part)
    log_exec = np.log10(exec_times)
    slope, intercept, r_value, _, _ = stats.linregress(log_part, log_exec)
    plt.loglog(num_part, exec_times, marker='o', linestyle='-', color=colors[i], label=f'{linenames[i]} (slope={slope:.3f})')

    # Reference Lines
    min_x = min(num_part)
    max_x = max(num_part)

    C_parallel = 10 ** intercept
    ref_y_min_parallel = C_parallel * (min_x ** slope)
    ref_y_max_parallel = C_parallel * (max_x ** slope)
    plt.plot([min_x, max_x], [ref_y_min_parallel, ref_y_max_parallel], linestyle='dotted', color=colors[i], alpha=0.5)

    plt.text(0.1, 0.95 - len(filenames) * 0.05 - i * 0.1, f'{linenames[i]} Slope: {slope:.3f}\nR^2: {r_value**2:.3f}', 
        transform=plt.gca().transAxes, fontsize=8,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.25'))

# Plot Labels and Legend
plt.xlabel('Number of Pixels (log scale)')
plt.ylabel('Execution Time (s) (log scale)')
plt.title('Execution Time vs Number of Pixels')

plt.legend(loc='upper left', borderaxespad=0., fontsize=8)
plt.grid(True)
plt.savefig(plot_name)
print(f"LogLog Plot saved as {plot_name}")
