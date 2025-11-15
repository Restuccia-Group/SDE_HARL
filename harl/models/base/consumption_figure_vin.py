import matplotlib.pyplot as plt
import numpy as np
#from google.colab import files as colab_files

labels = ['LC', 'Ours', 'SC', 'LC', 'Ours', 'SC']
x = np.arange(len(labels))
plt.rcParams["font.family"] = "Times New Roman"
# Figure (a) data
mobile = [0.136, 0.045, 0, 0.09, 0.025, 0]
comm   = [0, 0.0143, 0.2189, 0, 0.0143, 0.2189]
server = [0, 0.004, 0.01, 0, 0.004, 0.01]

# Figure (b) data
energy   = [0.952, 0.315, 0, 1.08, 0.3, 0]
transmit = [0, 896*1e-3,13680*1e-3, 0, 896*1e-3, 13680*1e-3]


# Setup plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.1))

# === Subplot (a): Latency ===
width = 0.6
ax1.bar(x, mobile, width, label='Local', color='deepskyblue', edgecolor='black', hatch='///')
ax1.bar(x, comm, width, bottom=mobile, label='Comm.', color='seagreen', edgecolor='black', hatch='\\\\')
ax1.bar(x, server, width, bottom=np.array(mobile)+np.array(comm), label='Server', color='darkorange', edgecolor='black')

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=24)
ax1.text(1, -0.037, 'Scenario 1', ha='center', va='top', fontsize=24, fontweight='bold')
ax1.text(4, -0.037, 'Scenario 2', ha='center', va='top', fontsize=24, fontweight='bold')
ax1.tick_params(axis='y', which='both', length=0, labelsize=14)
ax1.set_ylabel('Inference Times (ms)', fontsize=24)
ax1.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25), fontsize=24,
    handletextpad=0.4,    # horizontal space between legend handle and text
    columnspacing=0.6,    # horizontal space between columns
    borderpad=0.3,        # padding between legend content and border
    labelspacing=0.3)
#ax1.set_title("(a) Latency Breakdown", fontsize=14)
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

# === Subplot (b): Energy + Transmission ===
bar_width = 0.45
ax2b = ax2.twinx()

b1 = ax2.bar(x - bar_width/2, energy, width=bar_width, color='skyblue', edgecolor='black', hatch='///', label='Energy (mJ)')
b2 = ax2b.bar(x + bar_width/2, transmit, width=bar_width, color='darkorange', edgecolor='black', hatch='\\\\', label='Transmitted Data (KB)')

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=24)
ax2.text(1, -0.18, 'Scenario 1', ha='center', va='top', fontsize=24, fontweight='bold')
ax2.text(4, -0.18, 'Scenario 2', ha='center', va='top', fontsize=24, fontweight='bold')
ax2.tick_params(axis='y', which='both', length=0, labelsize=14)
ax2.set_ylabel('Energy Consumption (mJ)', fontsize=24)
ax2b.set_ylabel('Transmitted Data (KB)', fontsize=24)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
#ax2.set_title("(b) Energy and Transmission", fontsize=14)

ax2b.tick_params(axis='y', labelsize=14)

# Clean legend inside upper center of subplot (b)
lines = [b1[0], b2[0]]
labels_combined = ['Energy', ' Data']
ax2.legend(
    lines,
    labels_combined,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.26),
    ncol=2,
    fontsize=24,
    handletextpad=0.4,    # horizontal space between legend handle and text
    columnspacing=0.6,    # horizontal space between columns
    borderpad=0.3,        # padding between legend content and border
    labelspacing=0.3      # vertical space between legend entries
)

# Final layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.2, top=0.85)
plt.show()

# fig.savefig("latency_energy.png", dpi=300, bbox_inches='tight')
# colab_files.download("latency_energy.png")
