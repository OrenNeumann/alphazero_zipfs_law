import numpy as np
import matplotlib.pyplot as plt
import pickle
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import Figure

"""
Plot data from dkl sampling experiment
"""
print('loading data')
data_dir = r"/mnt/ceph/neumann/zipf/plot_data/sampling_counters/oware/s_3_2/"
with open(data_dir + "played.pkl", "rb") as input_file:
    played = pickle.load(input_file)
with open(data_dir + "sampled.pkl", "rb") as input_file:
    sampled = pickle.load(input_file)

print('done loading')
print(len(played))
print(len(sampled))


font = 18 - 2
font_num = 16 - 2

fig = Figure(x_label='State rank',
             y_label='Frequency',
             text_font=font,
             number_font=font_num,
             legend=True)

print('plotting zipfs law')
fig.preamble()
freq = np.array([item[1] for item in played.most_common()])

x = np.arange(len(freq)) + 1
plt.scatter(x, np.array(freq), s=40 / (10 + x), label='played')

freq = np.array([item[1] for item in sampled.most_common()])

x = np.arange(len(freq)) + 1
plt.scatter(x, np.array(freq), s=40 / (10 + x), label='sampled')

plt.xscale('log')
plt.yscale('log')
fig.epilogue()
fig.save('zipfs_law')

print('plotting turns')

with open(data_dir + "turns_played.pkl", "rb") as input_file:
    turns = pickle.load(input_file)


print(len(turns))
for key, count in played.items():
    turns[key] /= count


fig = Figure(fig_num=2,
             x_label='State rank',
             y_label='Turn',
             text_font=font,
             number_font=font_num)

fig.preamble()
y = sort_by_frequency(data=turns, counter=played)
x = np.arange(len(y)-1) + 1
plt.scatter(x, y[1:], s=40 * 3 / (10 + x), alpha=1, color='green')
plt.xscale('log')
plt.yscale('log')
fig.epilogue()
fig.save('turns_played')


with open(data_dir + "turns_sampled.pkl", "rb") as input_file:
    turns = pickle.load(input_file)
print(len(turns))
for key, count in sampled.items():
    turns[key] /= count


fig.fig_num = 3
fig.preamble()
y = sort_by_frequency(data=turns, counter=sampled)

x = np.arange(len(y)-1) + 1
plt.scatter(x, y[1:], s=40 * 3 / (10 + x), alpha=1, color='green')
plt.xscale('log')
plt.yscale('log')
fig.epilogue()
fig.save('turns_sampled')
