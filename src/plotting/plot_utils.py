import numpy as np
from typing import Optional

def incremental_bin(bin_max: int) -> np.ndarray:
    """
    Creates bin limits that expand exponentially.
    Bin n has a width of:
    n^(1+0.02*n)
    It's a good compromise between high detail and low noise.
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


def aligned_title(ax, title: str, font: int):
    """ Align the title to the left."""
    bbox = ax.get_yticklabels()[-1].get_window_extent()
    x,_ = ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
    ax.set_title(title, ha='left',x=x,fontsize=font)


def gaussian_average(y, sigma: float = 0.25, cut_tail: bool = False, mask: Optional[np.ndarray] = None):
    """ Smooth y by averaging it with a log-scale gaussian kernel.
        Giving a mask will ignore the values that are False in the mask.
        If cut_tail is True, the tail of the distribution will be cut 
         off by 2*sigma, to ignore tail-abberations."""
    ranks = np.arange(len(y))+1
    if mask is not None:
        ranks = ranks[mask]
        y = y[mask]
    x_ranks = np.arange(len(y))+1
    y_smooth = np.zeros_like(y)
    if cut_tail:
        y_smooth = np.zeros(int(len(y)/10**(2*sigma)))
        x_ranks =  np.copy(ranks[:int(len(y)/10**(2*sigma))])
    for i,r in enumerate(x_ranks):
        kernel = np.exp(-0.5 * ((np.log10(ranks/r)) / sigma) ** 2)
        y_smooth[i] = np.sum(y * kernel) / np.sum(kernel)
    if mask is not None:
        return y_smooth, x_ranks
    return y_smooth
