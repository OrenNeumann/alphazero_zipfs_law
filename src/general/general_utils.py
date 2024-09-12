import numpy as np
import yaml
import time


def models_path() -> str:
    with open("src/config/paths.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['paths']['models_dir']


def game_path(env: str) -> str:
    with open("src/config/paths.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['game_paths'][env]


def action_string(game: str) -> str:
    with open("src/config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['action_formats'][game]


def training_length(game: str) -> int:
    """ A rough estimate of the number of games (per actor) needed to train a model."""
    with open("src/config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['approx_training_length'][game]


def fit_power_law(freq: list[float],
                  up_bound: int,
                  low_bound: int,
                  full_equation: bool = False,
                  name: str = '',
                  min_x: int = 1,
                  max_x: int = 2 * 10 ** 6) -> tuple[np.ndarray, np.ndarray, str]:
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


class Timer:
    """ A time-keeping object.
        Initialize when you want to start the timer,
        and call 'stop' when you want to get the time passed."""

    def __init__(self):
        self.start = time.time()
        self.end = None

    def go(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()
        print('Runtime: %.2f hours.' % ((self.end - self.start) / 3600))
