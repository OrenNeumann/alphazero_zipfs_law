import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import numpy as np


class Figure(object):
    """
    Class for plotting a figure.

    Args:
        fig_num (int): The figure number.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the figure.
        text_font (int): The font size for the labels.
        number_font (int): The font size for the numbers.
        legend (bool): Whether to show the legend.

    Methods:
        preamble(): Set up the figure.
        epilogue(): Finalize the figure.
        save(name): Save the figure.
    """

    def __init__(self,
                 fig_num=1,
                 x_label='',
                 y_label='',
                 title='',
                 text_font=18,
                 number_font=16,
                 legend=False):
        self.fig_num = fig_num
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.text_font = text_font
        self.number_font = number_font
        self.legend = legend

    def preamble(self, fig_aspect=0.6):
        """ Call axis scaling commands after calling this."""
        w, h = plt.figaspect(fig_aspect)
        plt.figure(self.fig_num, figsize=(w, h))
        plt.style.use(['grid'])
        plt.clf()

    def epilogue(self):
        if self.x_label != '':
            plt.xlabel(self.x_label, fontsize=self.text_font)
        if self.y_label != '':
            plt.ylabel(self.y_label, fontsize=self.text_font)
        if self.title != '':
            plt.title(self.title, fontsize=self.text_font)
        if self.legend:
            plt.legend(fontsize=self.text_font - 3)
        plt.xticks(fontsize=self.number_font)
        plt.yticks(fontsize=self.number_font)
        plt.tight_layout()

    @staticmethod
    def save(name):
        plt.savefig('plots/' + name + '.png', dpi=900)


class BarFigure(Figure):
    """
    Class for bar figures (figures with a colorbar on the side).

    Extra Args:
        par (list or array): The parameter values for the bar.
    Extra Methods:
        colorbar_colors(): Calculate the color scale for the colorbar.
    """

    def __init__(self, par, colormap='viridis', **kwargs):
        super().__init__(**kwargs)
        self.cmap = colormap
        self.par = par

    def preamble(self, **kwargs):
        super().preamble(**kwargs)
        # ax = plt.gca()
        norm = matplotlib.colors.LogNorm(vmin=self.par.min(), vmax=self.par.max())
        # create a scalarmappable from the colormap
        sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap(self.cmap), norm=norm)
        cbar = plt.colorbar(sm)
        cbar.ax.tick_params(labelsize=self.number_font)
        cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=self.text_font)

    def colorbar_colors(self):
        """ calculate colorbar colors."""
        log_par = np.log(self.par)
        color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())
        return color_nums


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


def aligned_title(ax, title,font):
    """ Align the title to the left."""
    bbox = ax.get_yticklabels()[-1].get_window_extent()
    x,_ = ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
    ax.set_title(title, ha='left',x=x,fontsize=font)


def gaussian_average(y, sigma=0.25, cut_tail=False, mask=None):
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
