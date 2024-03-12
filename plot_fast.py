
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open(r"/home/oren/zipf/plot_data/erase/played_owar.pkl", "rb") as input_file:
    played = pickle.load(input_file)
with open(r"/home/oren/zipf/plot_data/erase/sample_owar.pkl", "rb") as input_file:
    sampled = pickle.load(input_file)

plt.figure(2, figsize=(10, 6))
freq = np.array([item[1] for item in played.most_common()])
x = np.arange(len(played)) + 1
plt.scatter(x, np.array(freq), s=40 / (10 + x), label=played)
freq = np.array([item[1] for item in sampled.most_common()])
x = np.arange(len(sampled)) + 1
plt.scatter(x, np.array(sampled), s=40 / (10 + x), label=sampled)
#played = list( c for  c in played if c >= 4)
#sampled = list( c for  c in sampled if c >= 4)

plt.ylabel('Frequency')
plt.xlabel('Board state number')
plt.xscale('log')
plt.yscale('log')
plt.title('Frequency of Board States')
#plt.legend()
#plt.tight_layout()
plt.savefig('plots/counts.png', dpi=900)