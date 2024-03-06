import matplotlib
import matplotlib.pyplot as plt
import scienceplots
import numpy as np


def figure_preamble(fig_num=1):
    w, h = plt.figaspect(0.6)
    plt.figure(fig_num, figsize=(w, h))
    plt.style.use(['grid'])
    plt.clf()


def figure_epilogue(x_label='', y_label='', title='', label_font=18, number_font=16,legend=False):
    if x_label is not '':
        plt.xlabel(x_label, fontsize=label_font)
    if y_label is not '':
        plt.ylabel(y_label, fontsize=label_font)
    if title is not '':
        plt.title(title, fontsize=label_font)
    if legend:
        plt.legend()
    plt.xticks(fontsize=number_font)
    plt.yticks(fontsize=number_font)
    plt.tight_layout()


def bar_figure_preamble(par, label_font=18, number_font=16, fig_num=1):
    # colorbar plot cargo-cult code
    figure_preamble(fig_num)
    # ax = plt.gca()
    norm = matplotlib.colors.LogNorm(vmin=par.min(), vmax=par.max())
    # create a scalarmappable from the colormap
    sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
    cbar = plt.colorbar(sm)
    cbar.ax.tick_params(labelsize=number_font)
    cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=label_font)


def colorbar_colors(par):
    """ calculate colorbar colors."""
    log_par = np.log(par)
    color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())
    return color_nums


class Figure:
    """
    A class representing a figure for plotting.

    Args:
        fig_num (int): The figure number.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the figure.
        label_font (int): The font size for the labels.
        number_font (int): The font size for the numbers.
        legend (bool): Whether to show the legend.

    Methods:
        figure_preamble(): Set up the figure.
        figure_epilogue(): Finalize the figure.
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

    def figure_preamble(self, fig_aspect=0.6):
        w, h = plt.figaspect(fig_aspect)
        plt.figure(self.fig_num, figsize=(w, h))
        plt.style.use(['grid'])
        plt.clf()

    def figure_epilogue(self):
        if self.x_label is not '':
            plt.xlabel(self.x_label, fontsize=self.text_font)
        if self.y_label is not '':
            plt.ylabel(self.y_label, fontsize=self.text_font)
        if self.title is not '':
            plt.title(self.title, fontsize=self.text_font)
        if self.legend:
            plt.legend(fontsize=self.text_font - 3)
        plt.xticks(fontsize=self.number_font)
        plt.yticks(fontsize=self.number_font)
        plt.tight_layout()

class BarFigure(Figure):
    """
    A class representing a bar figure.

    Args:
        par (list or array): The parameter values for the bar.
    """

    def __init__(self, par, fig_num=1, x_label='', y_label='', title='', text_font=18, number_font=16, legend=False):
        super().__init__(fig_num, x_label, y_label, title, text_font, number_font, legend)
        self.par = par

    def figure_preamble(self):
        super().figure_preamble()
        # ax = plt.gca()
        norm = matplotlib.colors.LogNorm(vmin=self.par.min(), vmax=self.par.max())
        # create a scalarmappable from the colormap
        sm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
        cbar = plt.colorbar(sm)
        cbar.ax.tick_params(labelsize=self.number_font)
        cbar.ax.set_ylabel('Parameters', rotation=90, fontsize=self.text_font)

    def colorbar_colors(self):
        """ calculate colorbar colors."""
        log_par = np.log(self.par)
        color_nums = (log_par - log_par.min()) / (log_par.max() - log_par.min())
        return color_nums
