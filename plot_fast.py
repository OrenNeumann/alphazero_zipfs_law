import numpy as np
import matplotlib.pyplot as plt
import pickle
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import Figure

"""
Plot data from dkl sampling experiment
"""

data_dir = r"/mnt/ceph/neumann/zipf/plot_data/sampling_counters/oware/s_3_2/"
def load_data(name):
    with open(data_dir + name+ ".pkl", "rb") as input_file:
        return pickle.load(input_file)


font = 18 - 2
font_num = 16 - 2

fig = Figure(x_label='State rank',
             y_label='Frequency',
             text_font=font,
             number_font=font_num,
             legend=True)
fig.preamble()
print('plotting zipfs law ~~~~~~~~~~~~')

print('loading played')
counts = load_data("played")
print('length: ', len(counts))

print('plotting played')


x = np.arange(len(counts)) + 1
plt.scatter(x, np.array([item[1] for item in counts.most_common()]), 
            s=40 / (10 + x), label='played')

print('loading sampled')
counts = load_data("sampled")
print('length: ', len(counts))
print('plotting sampled')

x = np.arange(len(counts)) + 1
plt.scatter(x, np.array([item[1] for item in counts.most_common()]), 
            s=40 / (10 + x), label='sampled')

plt.xscale('log')
plt.yscale('log')
fig.epilogue()
fig.save('zipfs_law')

print('plotting turns ~~~~~~~~~~~~')

print('loading turns sampled')
turns = load_data("turns_sampled")
print('length: ', len(turns))
print('plotting turns sampled')

for key, count in counts.items():
    turns[key] /= count


fig = Figure(fig_num=2,
             x_label='State rank',
             y_label='Turn',
             text_font=font,
             number_font=font_num)

fig.preamble()
y = sort_by_frequency(data=turns, counter=counts)
x = np.arange(len(y)-1) + 1
plt.scatter(x, y[1:], s=40 * 3 / (10 + x), alpha=1, color='green')
plt.xscale('log')
plt.yscale('log')
fig.epilogue()
fig.save('turns_sampled')

del(y)
del(x)

print('loading turns played')
turns = load_data("turns_played")
print(len(turns))
counts = load_data("played")
print('plotting turns played')

print(len(turns))
for key, count in counts.items():
    turns[key] /= count


fig.fig_num = 3
fig.preamble()
y = sort_by_frequency(data=turns, counter=counts)

x = np.arange(len(y)-1) + 1
plt.scatter(x, y[1:], s=40 * 3 / (10 + x), alpha=1, color='green')
plt.xscale('log')
plt.yscale('log')
fig.epilogue()

fig.save('turns_played')