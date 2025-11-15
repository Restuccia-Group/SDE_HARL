import torch
import torch.nn as nn
from time import perf_counter as timer

# Accurate GPU timing using PyTorch events
def measure_gpu_time(func):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    func()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 1000  # seconds

# Build MLP
def build_block_mlp(input_dim, num_blocks, output_dim, hidden_dim=128):
    layers = []
    in_dim = input_dim
    for _ in range(num_blocks):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)

# Constants
number_size_obs = 138
batch_size = 10000
obs = torch.randn(batch_size, number_size_obs)
network_bandwidth_mbps = 500
ratios = [(0, 10), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (10, 0)]
BPP = [5.46, 1.57, 1.42, 1.25, 1.02, 0.98, 0]

NUM_RUNS = 10000
WARMUP = 10

# Adjusted power profiles (Watt)
CPU_POWER_EST = 25   # For higher load MLP
GPU_POWER_EST = 120  # For modern GPUs under full load

print("Split | Edge Time (ms) | Server Time (ms) | Transmit Time (ms) | Total Time (ms) | CPU Energy (mJ) | GPU Energy (mJ) | BW (bytes)")
print("-" * 130)

for idx, (edge_blocks, server_blocks) in enumerate(ratios):
    use_edge = edge_blocks > 0
    use_server = server_blocks > 0

    if use_edge:
        edge_model = build_block_mlp(number_size_obs, edge_blocks, 128).cpu()
    if use_server:
        input_dim = 128 if use_edge else number_size_obs
        server_model = build_block_mlp(input_dim, server_blocks, 10).cuda()

    # Warm-up
    for _ in range(WARMUP):
        with torch.no_grad():
            if not use_edge:
                z_gpu = obs.to("cuda")
            else:
                z = edge_model(obs)
                z_gpu = z.to("cuda") if use_server else z
            if use_server:
                _ = server_model(z_gpu)
                torch.cuda.synchronize()

    # Edge time
    if use_edge:
        edge_times = []
        for _ in range(NUM_RUNS):
            start = timer()
            z = edge_model(obs)
            end = timer()
            edge_times.append(end - start)
        avg_cpu_time = sum(edge_times) / NUM_RUNS
    else:
        z = obs
        avg_cpu_time = 0.0

    # Transmission
    if use_edge and use_server:
        bpp = BPP[idx]
        num_points = z.numel()
        total_bits = bpp * num_points
        transmit_time = total_bits / (network_bandwidth_mbps * 1e6)
        bandwidth_bytes = total_bits / 8
        z_gpu = z.to("cuda")
    elif not use_edge and use_server:
        bpp = BPP[idx]
        num_points = obs.numel()
        total_bits = bpp * num_points
        transmit_time = total_bits / (network_bandwidth_mbps * 1e6)
        bandwidth_bytes = total_bits / 8
        z_gpu = obs.to("cuda")
    else:
        transmit_time = 0.0
        bandwidth_bytes = 0.0
        z_gpu = None

    # GPU time
    if use_server:
        with torch.no_grad():
            gpu_times = []
            for _ in range(NUM_RUNS):
                gpu_times.append(measure_gpu_time(lambda: server_model(z_gpu)))
        avg_gpu_time = sum(gpu_times) / NUM_RUNS
    else:
        avg_gpu_time = 0.0

    # Energy Estimation
    cpu_energy_mJ = CPU_POWER_EST * avg_cpu_time * 1000  # mJ
    gpu_energy_mJ = GPU_POWER_EST * avg_gpu_time * 1000  # mJ
    total_time = avg_cpu_time + transmit_time + avg_gpu_time

    # Print
    print(f"{edge_blocks:<5} | {avg_cpu_time*1e3/batch_size:<14.4f} | {avg_gpu_time*1e3/batch_size:<15.4f} | {transmit_time*1e3/batch_size:<18.4f} | {total_time*1e3/batch_size:<15.4f} | {cpu_energy_mJ/batch_size:<15.6f} | {gpu_energy_mJ/batch_size:<15.6f} | {bandwidth_bytes/batch_size:.2f}")
