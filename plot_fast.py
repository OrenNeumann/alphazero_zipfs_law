import numpy as np
import matplotlib.pyplot as plt
import pickle
from src.data_analysis.data_utils import sort_by_frequency
from src.plotting.plot_utils import Figure

data_dir = r"/mnt/ceph/neumann/zipf/plot_data/erase/che_test/"
with open(data_dir + "played.pkl", "rb") as input_file:
    played = pickle.load(input_file)
with open(data_dir + "sampled.pkl", "rb") as input_file:
    sampled = pickle.load(input_file)
with open(data_dir + "turns_played.pkl", "rb") as input_file:
    turns_played = pickle.load(input_file)
with open(data_dir + "turns_sampled.pkl", "rb") as input_file:
    turns_sampled = pickle.load(input_file)

print(len(played))
print(len(sampled))
print(len(turns_played))
print(len(turns_sampled))

print('done loading')
for key, count in played.items():
    turns_played[key] /= count
for key, count in sampled.items():
    turns_sampled[key] /= count

font = 18 - 2
font_num = 16 - 2

fig = Figure(x_label='State rank', text_font=font, number_font=font_num)


def new_fig(y_label):
    fig.fig_num += 1
    fig.preamble()

    fig.y_label = y_label


print('plotting zipfs law')
new_fig('Frequency')
freq = np.array([item[1] for item in played.most_common()])
#freq = freq[:10**4]
x = np.arange(len(freq)) + 1
plt.scatter(x, np.array(freq), s=40 / (10 + x), label=played)

freq = np.array([item[1] for item in sampled.most_common()])
#freq = freq[:10**4]
x = np.arange(len(freq)) + 1
plt.scatter(x, np.array(freq), s=40 / (10 + x), label=sampled)
#plt.legend(loc="upper right")

if fig.x_label != '':
    plt.xlabel(fig.x_label, fontsize=fig.text_font)
if fig.y_label != '':
    plt.ylabel(fig.y_label, fontsize=fig.text_font)
if fig.title != '':
    plt.title(fig.title, fontsize=fig.text_font)
if fig.legend:
    plt.legend(fontsize=fig.text_font - 3)
plt.xticks(fontsize=fig.number_font)
plt.yticks(fontsize=fig.number_font)
plt.xscale('log')
plt.yscale('log')

#fig.epilogue()
fig.save('zipfs_law')

print('plotting turns')
new_fig('Turn')
y = sort_by_frequency(data=turns_played, counter=sampled)
x = np.arange(len(y)) + 1
plt.scatter(x, y, s=40 * 3 / (10 + x), alpha=1, color='green')
fig.epilogue()
fig.save('turns_played')

new_fig('Turn')
y = sort_by_frequency(data=turns_sampled, counter=sampled)
x = np.arange(len(y)) + 1
plt.scatter(x, y, s=40 * 3 / (10 + x), alpha=1, color='green')
fig.epilogue()
fig.save('turns_sampled')
