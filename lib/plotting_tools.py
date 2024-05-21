def adjust_plot_1d(fig, ax, plot_args=None):
    if not plot_args:
        plot_args = {}

    for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(1.5)
          ax.spines[axis].set_color('0.0')

    y_scale_in_log = plot_args.get('y_axis_in_log', False)
    if(y_scale_in_log):
        ax.set_yscale('log')

    ax.tick_params(axis='both', which='both', width=1.5, colors='0.0', labelsize=18)
    ax.yaxis.set_ticks_position('both')
    ax.set_ylabel(plot_args.get('ylabel', 'pdf'), fontsize=20)
    ax.set_xlabel(plot_args.get('xlabel', 'var 1'), fontsize=20)
    ax.set_ylim(plot_args.get('ylim', [0, 1]))
    ax.set_xlim(plot_args.get('xlim', [0, 1]))
    ax.legend()
