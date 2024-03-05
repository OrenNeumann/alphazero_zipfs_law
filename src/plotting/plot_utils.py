import matplotlib
import matplotlib.pyplot as plt
import scienceplots

# this should be a class

def figure_preamble(fig_num=1):
    w, h = plt.figaspect(0.6)
    plt.figure(fig_num, figsize=(w, h))
    plt.style.use(['grid'])
    plt.clf()


def figure_epilogue(x_label='', y_label='', title='', label_font=18, number_font=16):
    if x_label is not '':
        plt.xlabel(x_label, fontsize=label_font)
    if y_label is not '':
        plt.ylabel(y_label, fontsize=label_font)
    if title is not '':
        plt.title(title, fontsize=label_font)
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
