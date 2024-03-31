import matplotlib.pyplot as plt


def game_turns():
    tf =10
    # Create figure and subplots
    fig = plt.figure(figsize=(12, 6))

    # Define grid for subplots
    # Divide the figure into 3 columns and 2 rows
    # The 5th subplot (index 4) spans 2 rows and 1 column
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax5 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

    square_plots = [ax1, ax2, ax3, ax4]
    colors = ['blue', 'purple', 'green', 'olive']
    for i, ax in enumerate(square_plots):
        ax.scatter([1, 2, 3], [4, 5, 6], color=colors[i],label='Line ' + str(i + 1))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc="upper left")
        ax.tick_params(axis='both', which='major', labelsize=tf)


    # Add axis labels to each subplot
    ax1.set_ylabel('Y Label 1',fontsize=tf)
    ax3.set_xlabel('X Label 3',fontsize=tf)
    ax3.set_ylabel('Y Label 3',fontsize=tf)
    ax4.set_xlabel('X Label 4',fontsize=tf)

    ax5.plot([1, 2, 3], [4, 5, 6])
    ax5.set_xlabel('X Label 5',fontsize=tf+4)
    ax5.set_ylabel('Y Label 5',fontsize=tf+4)


    ax5.tick_params(axis='both', which='major', labelsize=tf+4)

    # Set titles for each subplot
    bbox = ax1.get_yticklabels()[-1].get_window_extent()
    x,_ = ax1.transAxes.inverted().transform([bbox.x0, bbox.y0])
    ax1.set_title(r'$\bf{a.}$ Turn distribution', ha='left',x=x,fontsize=tf+4)
    bbox = ax5.get_yticklabels()[-1].get_window_extent()
    x,_ = ax5.transAxes.inverted().transform([bbox.x0, bbox.y0])
    ax5.set_title(r'$\bf{b.}$ Turn ratios', ha='left',x=x,fontsize=tf+4)


    # Adjust layout to prevent overlapping of subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
