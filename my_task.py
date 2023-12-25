import numpy as np
from dimod import BinaryQuadraticModel
from neal import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
import time

# Define QUBO coefficients
h = np.array([-0.72, -0.507, 0.7, -0.322, -0.534, 1.098, -0.547, -0.182, -0.812, 0.298])
J = np.array([
    [-0.464, 0.536, 0.608, -0.438, 0.542, 0.016, -1.269, 1.704, -0.958, -0.385],
    [-2.473, 0.616, 0.106, 0.518, 0.44, -1.186, 1.904, -0.158, 0.033, 0.556],
    [-0.403, -0.095, 0.322, -0.811, 1.193, -1.328, -0.648, 0.86, -1.698, -0.454],
    [-0.685, -0.52, -0.604, 0.308, 2.272, -0.295, -1.246, 0.537, -0.011, -0.85],
    [-0.762, 0.784, 0.542, 0.502, -0.719, -0.182, -1.596, 0.036, -0.17, -0.618],
    [-0.303, 0.361, -1.61, 0.113, 0.206, -0.87, 0.051, 0.351, -1.068, 0.804],
    [-1.703, -1.472, -0.989, -0.226, -0.936, -0.236, 0.661, -0.255, -0.172, -0.259],
    [0.48, -0.055, 0.025, -0.433, -0.326, 2.225, 0.661, -0.259, -0.343, -1.497],
    [1.074, 0.595, -0.567, -1.007, 1.684, 0.141, 0.593, 0.351, -0.311, -0.607],
    [0.872, 0.992, 0.36, 0.23, -0.876, 0.379, 1.053, -1.759, -0.816, -0.152]
])

# Create binary variables
x = {f'x{i}': 0 for i in range(10)}

# Define QUBO model
model = BinaryQuadraticModel.from_numpy_matrix(J, offset=h.sum())

# Solve model with simulated annealing on CPU
sampler_cpu = SimulatedAnnealingSampler()
start_cpu = time.time()
sampleset_cpu = sampler_cpu.sample(model, num_reads=10, num_sweeps=1000)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# Use LeapHybridSampler from dwave.system for GPU SA
# Update 'api_key' with your actual API key
sampler_gpu = LeapHybridSampler(solver=dict(qpu=True, hybrid_mode="sweep"), token='YOUR_API_KEY')
start_gpu = time.time()
sampleset_gpu = sampler_gpu.sample(model, num_reads=10, num_sweeps=1000)
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

speedup = cpu_time / gpu_time
print(f"GPU was {speedup:.2f}x faster than CPU")

# Filter for feasible samples on CPU
feasible_sampleset_cpu = sampleset_cpu.filter(lambda x: x.energy <= model.energy(x.sample))
best_cpu = feasible_sampleset_cpu.first.sample

# Print solution and time on CPU
print("CPU Solution:", best_cpu)
print("CPU Exact energy:", model.energy(best_cpu))
print("CPU Runtime:", cpu_time)

# Filter for feasible samples on GPU
feasible_sampleset_gpu = sampleset_gpu.filter(lambda x: x.energy <= model.energy(x.sample))
best_gpu = feasible_sampleset_gpu.first.sample

# Print solution and time on GPU
print("GPU Solution:", best_gpu)
print("GPU Exact energy:", model.energy(best_gpu))
print("GPU Runtime:", gpu_time)
