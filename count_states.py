import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
from data_analysis.game_data_analysis import process_games, get_value_estimators


"""
Count board states played from actor logfiles of AlphaZero agents.
"""

# Choose game type:
game_num = 3
games = ['connect4', 'pentago', 'oware', 'checkers']

env = games[game_num]
path = '/mnt/ceph/neumann/alphazero/scratch_backup/models/'
data_paths = {'connect4': 'connect_four_10000/q_0_0',
              'pentago': 'pentago_t5_10000/q_0_0',
              'oware': 'oware_10000/q_1_0',
              'checkers': 'checkers/q_6_0'}
path += data_paths[env]
print('Collecting '+env+' games:')

# Process all games
# board_counter, _ = process_games(env, path)
board_counter, information = process_games(env, path, save_serial=True,
                                           save_turn_num=True, max_file_num=39)
serial_states = information['serials']
turn_numbers = information['turn_nums']

# Sort by frequency
board_freq = sorted(board_counter.items(), key=lambda x: x[1], reverse=True)

# Extract the keys and the frequencies
# keys = [item[0] for item in board_freq]
freq = [item[1] for item in board_freq]

# Fit a power-law
if env == 'connect4':
    low = 5 * 10 ** 2  # lower fit bound
    up = 10 ** 5  # upper fit bound
elif env == 'pentago':
    low = 2 * 10 ** 3  # lower fit bound
    up = 2 * 10 ** 5  # upper fit bound
else:
    low = 10**2
    up = int(len(freq)/10**2)
x_nums = np.arange(up)[low:]
[m, c] = np.polyfit(np.log10(np.arange(up)[low:] + 1), np.log10(freq[low:up]), deg=1, w=2 / x_nums)
equation = '10^{c:.2f} * n^{m:.2f}'

# Plot
plt.figure(figsize=(10, 6))
n_points = len(freq) #5 * 10 ** 6  # Display top n_points

#x_fit = np.array([1, n_points])
x_fit = np.array([low, up])
y_fit = 10 ** c * x_fit ** m

x = np.arange(n_points) + 1
plt.scatter(x, freq[:n_points], s=40 / (10 + x))
plt.plot(x_fit, y_fit, color='red', linewidth=1.5, label=equation.format(c=c, m=m))
plt.ylabel('Frequency')
plt.xlabel('Board state number')
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=n_points)
plt.title('Frequency of Board States')
plt.legend()
plt.tight_layout()
plt.savefig('zipf_distribution.png', dpi=900)
plt.show()

# Value loss analysis:
print('loss part...')
path_model = '/mnt/ceph/neumann/alphazero/scratch_backup/models/connect_four_10000/q_6_0/'
solver_value, model_value = get_value_estimators(env, path_model)
# losses = {key: loss(serial) for key, serial in serial_states.items()}
losses = {}
model_values = {}
solver_values = {}
for key, serial in tqdm(serial_states.items(), desc="Estimating model state values"):
    model_values[key] = model_value(serial)
for key, serial in tqdm(serial_states.items(), desc="Estimating solver state values"):
    solver_values[key] = solver_value(serial)
    losses[key] = (solver_values[key] - model_values[key]) ** 2

with open('saved_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)
with open('solver_values.pkl', 'wb') as f:
    pickle.dump(solver_values, f)
# with open('solver_values.pkl', "rb") as input_file:
#   solver_values = pickle.load(input_file)

# losses = {key: loss(serial) for key, serial in tqdm(serial_states.items(), desc='Calculating loss')}


"""
# plotting. 
y_model = []
y_solver = []
y_losses = []
x = []
# sort values by descending frequency:
for entry in board_counter.most_common():
    key = entry[0]
    y_model.append(model_values[key])
    y_solver.append(solver_values[key])
    y_losses.append((solver_values[key] - model_values[key])**2)
    x.append(entry[1])

# split win/loss/draw datasets:
win, loss, draw = [], [], []
for i in range(len(x)):
    if y_solver[i] == 1:
        win.append(i)
    elif y_solver[i] == -1:
        loss.append(i)
    elif y_solver[i] == 0:
        draw.append(i)
    else:
        raise Exception('invalid solver value: '+ str(y_solver[i]))
        
plt.figure(1)
plt.scatter(np.array(win) + 1, np.array(y_model)[win], s=4, alpha=0.3)
plt.scatter(np.array(loss) + 1, np.array(y_model)[loss], s=4, alpha=0.3)
plt.scatter(np.array(draw) + 1, np.array(y_model)[draw], s=4, alpha=0.3)

plt.xscale('log')
plt.xlabel('state rank')
plt.ylabel('model prediction')

# plotting cumulative average of loss
# the cumulative average makes a lot of sense, in logscale this average is 
# always dominated by the latest entries and accurately represents them.
# if the model started being good at labeling high-rank states,
# this curve should go down.
# But, what we see is loss going monotonically up => less frequent
# states always have higher loss on average.
plt.figure(2)
y = np.cumsum(np.array(y_losses)) / (np.arange(len(x)) + 1)
plt.scatter(np.arange(len(x)) + 1, y, s=4, alpha=0.3)
plt.xscale('log')
plt.xlabel('state rank')
plt.ylabel('Cumulative average of the loss')
plt.title('Value loss change with state rank')
"""

""" #not useful
# plot the average loss across frequency. higher freq. states have lower loss.
freq_values = set(x)
average_loss = []
x_frequencies = []
for frequency in tqdm(freq_values, desc="Averaging loss"):
    x_frequencies.append(frequency)
    indices = [ind for ind, entry in enumerate(x) if entry==frequency]
    average_loss.append(sum(np.array(y_losses)[indices])/len(indices))
    
plt.figure(2)
plt.scatter(x_frequencies, average_loss, s=4)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('state frequency')
plt.ylabel('model loss (averaged)')
"""

""" #crap
# plot colors on zipf curve: 

# get indices of states with low loss:
learned = np.arange(len(x))[np.array(y_losses) < 0.25**2]
not_learned = np.arange(len(x))[np.array(y_losses) > 0.25**2]

# Plot
plt.figure(figsize=(10, 6))
n_points = len(freq) #5 * 10 ** 6  # Display top n_points
plt.scatter(not_learned + 1, np.array(freq)[not_learned], s=400/(10+not_learned), alpha=0.3)
plt.scatter(learned + 1, np.array(freq)[learned]*10, s=400/(10+learned), alpha=0.3)
plt.ylabel('Frequency')
plt.xlabel('Board state number')
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=n_points)
plt.title('Frequency of Board States (Connect Four, T=0.1)')
plt.tight_layout()
plt.show()

"""

"""
# calculating % of predictions above a certain threshold in each category.
# while calculating the mean is crap, this metric is promising.
# winning states have a higher % of values above 0 than losing states.

threshold = 0.0
def f(arr):
    s = 0
    for i in arr:
        if i > threshold:
            s+=1
    print(s/len(arr))
print('win percentage:')
f(np.array(y_model)[win])
print('loss percentage:')
f(np.array(y_model)[loss])



# for thresholds in the range [-0.01,0.07] there are ~7.5% more win states than loss states.

def f(arr, threshold):
    s = 0
    for i in arr:
        if i > threshold:
            s+=1
    return s/len(arr)

for t in range(10):
    thr = 0.01 + t/100
    a = f(np.array(y_model)[win],thr) - f(np.array(y_model)[loss],thr)
    print('thr: ' + str(thr))
    print(a)
    
    
# should make a histogram, where each model-value range shows the % of win/loss states.
# my guess is there are some ranges where the difference peaks
"""
