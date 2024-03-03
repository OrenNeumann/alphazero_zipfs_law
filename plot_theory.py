import numpy as np
import matplotlib.pyplot as plt


"""
sigma = 1
mu = 0
def gaussian(x):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))

values = np.arange(-10,10,0.0001)

probs = gaussian(values)

n = np.arange(1,len(values))

# sort n by probs:
probs, n_sorted = zip(*sorted(zip(probs, n)))

#plt.loglog(n[::-1],np.array(probs)/min(probs),'*')
plt.xscale('log')
plt.yscale('log')
plt.scatter(n[::-1],np.array(probs)/min(probs), color='dodgerblue', s=4, alpha=0.3)
plt.xlabel('rank')
plt.ylabel('frequency')

"""


depth = 8
b_factor = 7

l = 0
# l =  (b_factor**(depth+1)-1)/(b_factor-1)  -1
for i in range(depth):
    l+= b_factor**(i+1)
freq = np.zeros(l)
ind = 0
for d in range(depth):
    count = b_factor**(d+1)
    freq[ind:ind+count] = 1./count
    ind = ind+count
# divide by branching factor:
#freq = freq/sum(freq)
freq = freq/freq.min()
x = np.arange(l) + 1


w, h = plt.figaspect(0.7) #0.6
plt.figure(2, figsize=(w, h))
plt.style.use(['grid'])

font = 18-2
font_num = 16-2


#plt.scatter(x,freq, color='dodgerblue', s=2*40 / (10 + x), alpha=1)
#[m, c] = np.polyfit(np.log10(x), np.log10(freq), deg=1, w=2 / np.sqrt(x))
m =-1.
c = np.log10(freq.max()) + 0.5
exp = str(round(-m, 2))
const_exp = str(round(-c / m, 2))
# equation = r'$ \left( \frac{n}{ 10^{'+const_exp+r'} } \right) ^ {\mathbf{'+exp+r'}}$'
equation = r' $\alpha = ' + exp + '$'

# Plot
upper_bound = int(2.5*10**6)
y_fit = 10 ** c * x[:upper_bound] ** m

plt.scatter(x,freq, color='dodgerblue', s=2*40 / (10 + x), alpha=1)
plt.plot(x[:upper_bound], y_fit, color='black', linewidth=1.3, alpha=0.7, label=equation)

plt.xscale('log')
plt.yscale('log')
plt.title('Random games, theory', fontsize=font)
plt.ylabel('Frequency', fontsize=font)
plt.xlabel('Board state rank', fontsize=font)
plt.xticks(fontsize=font_num)
plt.yticks(fontsize=font_num)

plt.legend(fontsize=font - 3, framealpha=1)

plt.tight_layout()
plt.savefig('plots/theory.png', dpi=900)
