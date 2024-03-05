import numpy as np
import yaml


def models_path():
    with open("src/config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['paths']['models_dir']


def game_path(game:str):
    with open("src/config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['game_paths'][game]


def fit_power_law(freq,
                  up_bound,
                  low_bound,
                  full_equation=False,
                  name='',
                  min_x=1,
                  max_x=2 * 10 ** 6):
    x_nums = np.arange(up_bound)[low_bound:] + 1
    y_nums = freq[low_bound:up_bound]
    [m, c] = np.polyfit(np.log10(x_nums), np.log10(y_nums), deg=1, w=2 / x_nums)
    exp = str(round(-m, 2))
    if full_equation:
        const_exp = str(round(-c / m, 2))
        equation = r'$ \left( \frac{n}{ 10^{' + const_exp + r'} } \right) ^ {\mathbf{' + exp + r'}}$'
    else:
        if name == '':
            equation = r'$\alpha = ' + exp + '$'
        else:
            equation = name + r', $\alpha = ' + exp + '$'

    x_fit = np.array([min_x, max_x])
    y_fit = 10 ** c * x_fit ** m
    return x_fit, y_fit, equation


def fit_logaritm(freq,
                 up_bound,
                 low_bound,
                 n_points=None):
    # not used
    x_nums = np.arange(up_bound)[low_bound:] + 1
    y_nums = freq[low_bound:up_bound]
    [m, c] = np.polyfit(np.log10(x_nums), y_nums, deg=1, w=2 / x_nums)
    slope = str(round(m, 2))
    const = str(round(c, 2))
    equation = r'$ {' + slope + r'} \cdot \log_{10} n + {' + const + r'}$'

    if n_points is None:
        # Plot fit up to y=1:
        n_points = 10 ** ((1 - c) / m)

    x_fit = np.power(10, np.linspace(0.0, np.log10(n_points), num=100))
    y_fit = m * np.log10(x_fit) + c
    return x_fit, y_fit, equation


def incremental_bin(bin_max):
    """
    Creates bin limits that expand exponentially.
    Bin n has a width of:
    n^(1+0.02*n)
    I found that's a good compromise between high detail and low noise.
    """
    bins = [1]
    alpha = 1
    for n in range(bin_max):
        new_val = bins[-1] + (n + 1) ** alpha
        alpha += 0.02
        if new_val >= bin_max:
            bins.append(bin_max)
            break
        bins.append(new_val)
    return np.array(bins)
