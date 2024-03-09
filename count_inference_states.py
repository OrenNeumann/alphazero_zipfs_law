from src.data_analysis.state_frequency.state_counter import StateCounter, RandomGamesCounter
import matplotlib.pyplot as plt
import numpy as np

env = 'connect_four'
random = True
temp = 0.25
#env = 'pentago'
if env == 'connect_four':
    print('Collecting Connect Four games:')
    if temp == 1:
        path = '/mnt/ceph/neumann/zipfs_law/inference_matches/' \
               'connect_fourq_6_0q_6_1/' \
               'checkpoint_10000temp1/log-matches'
        filename = 'q_6_0q_6_1_temp1'
    elif temp == 0.25:
        path = '/mnt/ceph/neumann/zipfs_law/inference_matches/' \
               'connect_fourq_6_0q_6_1/' \
               'checkpoint_10000/log-matches'
        filename = 'q_6_0q_6_1_temp0p25'
    else:
        raise ValueError('only supports T=1,0.25')
else:
    print('Collecting Pentago games:')
    raise NameError('not implemented for Pentago yet')
    #path = '/mnt/ceph/neumann/alphazero/scratch_backup/models/pentago_t5_10000/q_0_0/log-actor'

if not random:
    # Process all games
    state_counter = StateCounter(env=env)
    state_counter.collect_data(path=path, max_file_num=80) # max_file_num=40)
else:
    # Generate random games:
    state_counter = RandomGamesCounter(env=env)
    state_counter.collect_data(n=3*25_000*80)
    filename = 'random2'

# Sort by frequency
board_freq = sorted(state_counter.frequencies.items(), key=lambda x: x[1], reverse=True)

# Extract the keys and the frequencies
# keys = [item[0] for item in board_freq]
freq = [item[1] for item in board_freq]
freq = np.array(freq)
save_path = '/mnt/ceph/neumann/AZ_new/count_states/plot_data/' + env + '/test_' + filename
np.save(save_path + '.npy', freq)
print('Data saved successfully.')

# Fit a power-law
if env == 'connect4':
    low = 5 * 10 ** 2  # lower fit bound
    up = 10 ** 5  # upper fit bound
else:
    low = 2 * 10 ** 3  # lower fit bound
    up = 2 * 10 ** 5  # upper fit bound
x_nums = np.arange(up)[low:]
y_nums = freq[low:up]
# Fit power law with decaying weights 'w' to avoid bias towards high board numbers:
[m, c] = np.polyfit(np.log10(x_nums), np.log10(y_nums), deg=1, w=2 / x_nums)
#[m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(freq[low:up]), 1)
#equation = '10^{c:.2f} * n^{m:.2f}'
equation = r'$10^{'+'{c:.2f}'.format(c=c)+r'} \cdot n^{\bf{'+'{m:.2f}'.format(m=m)+r'}}$'

# Plot
plt.figure(figsize=(10, 6))
n_points = len(freq) #5 * 10 ** 6  # Display top n_points

x_fit = np.array([1, n_points])
y_fit = 10 ** c * x_fit ** m

plt.scatter(np.arange(n_points) + 1, freq[:n_points], s=4, alpha=0.3)
#plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation.format(c=c, m=m))
plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation)
plt.ylabel('Frequency')
plt.xlabel('Board state number')
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=n_points)
plt.title('Frequency of Board States (Connect Four, T=0.1)')
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()


# add together 3 dicts:
#a={k: board_counter.get(k, 0) + board_counter2.get(k, 0) + board_counter3.get(k, 0) for k in set(board_counter) | set(board_counter2) | set(board_counter3)}
# Weird: the intersection of 3 agent pairs (6-6,6-0,0-0) only has 31 (!!!) boards, compared to the total
# number of boards, 141,559. I wonder if 2 pairs of similar strengths have a larger intersection.