import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.io import loadmat
# import pickle
# import time


def get_ge_from_success_info(sinfo, nr_traces_vec):
    """
    Returns guessing entropy data from the given success info structure.

    Parameters:
    - sinfo: Structured array or nested numpy array containing depth data for both
      ".avg" and ".joint" scores.
    - nr_traces_vec: List or numpy array of length nr_test_groups, with the number of
      attack traces used in the experiments.

    Returns:
    - g: Dictionary with two fields:
        - 'avg': Guessing entropy data from the ".avg" scores in sinfo.
        - 'joint': Guessing entropy data from the ".joint" scores in sinfo.
        Both 'avg' and 'joint' are numpy arrays of length nr_test_groups.
    """

    d_avg = sinfo['depth']['avg']
    d_joint = sinfo['depth']['joint']
    nr_test_groups = len(nr_traces_vec)

    g = {
        'avg': np.zeros(nr_test_groups),
        'joint': np.zeros(nr_test_groups)
    }

    # Compute the mean guessing entropy for all data
    for k in range(nr_test_groups):
        nr_iter = d_avg[f'group{k+1}'].shape[1]
        avg_sum = 0
        joint_sum = 0

        for j in range(nr_iter):
            avg_sum += np.log2(np.mean(d_avg[f'group{k+1}'][:, j]))
            joint_sum += np.log2(np.mean(d_joint[f'group{k+1}'][:, j]))
            # in the original code, the values for 
            # sinfo['depth']['avg'][f'group{k+1}'][:, j]
            # decrease as k increases, which is not the case here
            # TODO: figure out why and fix
            # print(d_avg[f'group{k+1}'][:, j])
            # print(d_joint[f'group{k+1}'][:, j])

        g['avg'][k] = avg_sum / nr_iter
        g['joint'][k] = joint_sum / nr_iter
        # print(g)

    # print(g)
    return g

def get_line_properties_templates(uid, style):
    """
    Returns line properties for plots of template attack results.
    
    Parameters:
    - uid: Integer uniquely identifying the desired line properties (from 1 to 6).
    - style: String specifying the kind of style ('normal' or 'fancy').

    Returns:
    - slines: Dictionary containing line properties ('Color', 'LineStyle', 'LineWidth', 'Marker').
    """

    if style == 'normal':
        if uid == 0:
            slines = {'Color': 'm', 'LineStyle': '-', 'LineWidth': 4, 'Marker': 'none'}
        elif uid == 1:
            slines = {'Color': 'c', 'LineStyle': '-.', 'LineWidth': 4, 'Marker': 'none'}
        elif uid == 2:
            slines = {'Color': 'g', 'LineStyle': '-', 'LineWidth': 4, 'Marker': 'none'}
        elif uid == 3:
            slines = {'Color': 'k', 'LineStyle': '-.', 'LineWidth': 4, 'Marker': 'none'}
        elif uid == 4:
            slines = {'Color': 'b', 'LineStyle': '-', 'LineWidth': 4, 'Marker': 'none'}
        elif uid == 5:
            slines = {'Color': 'r', 'LineStyle': '-.', 'LineWidth': 4, 'Marker': 'none'}
        else:
            raise ValueError('uid not supported')

    elif style == 'fancy':
        if uid == 0:
            slines = {'Color': 'm', 'LineStyle': '-', 'LineWidth': 1, 'Marker': 'o'}
        elif uid == 1:
            slines = {'Color': 'c', 'LineStyle': '--', 'LineWidth': 1, 'Marker': '+'}
        elif uid == 2:
            slines = {'Color': 'g', 'LineStyle': '-.', 'LineWidth': 1, 'Marker': '*'}
        elif uid == 3:
            slines = {'Color': 'k', 'LineStyle': '-', 'LineWidth': 1, 'Marker': '.'}
        elif uid == 4:
            slines = {'Color': 'b', 'LineStyle': '--', 'LineWidth': 1, 'Marker': 'x'}
        elif uid == 5:
            slines = {'Color': 'r', 'LineStyle': '-.', 'LineWidth': 1, 'Marker': 's'}
        else:
            raise ValueError('uid not supported')

    else:
        raise ValueError('Unknown style')

    return slines

def make_figures_ge(G, nr_traces_vec, rpath, rprefix, title_ge, slegend, font_size,
                    slines, options, yrange):
    """
    Creates figures for guessing entropy data.

    Parameters:
    - G: 2D numpy array of shape (nr_exp, nr_test_groups).
    - nr_traces_vec: List or numpy array of number of test traces (x-axis).
    - rpath: Path to save the figures.
    - rprefix: Prefix for saved figure filenames.
    - title_ge: Title for the figures.
    - slegend: List of legends for each experiment.
    - font_size: Font size for text in the plot.
    - slines: List of dictionaries containing line properties.
    - options: String containing plot options (e.g., 'y', 'g', 'L').
    - yrange: Tuple specifying y-axis limits.
    """
    nr_exp = G.shape[0]

    if font_size is None:
        font_size = 24

    nr_test_groups = len(nr_traces_vec)
    if G.shape[1] != nr_test_groups:
        raise ValueError('Incompatible nr_test_groups')

    xl_str = '$n_a$ (log axis)'
    font_small = 14

    fig_size = (10.24, 8.68) if 'L' in options else (6.4, 4.8)

    # Plot guessing entropy data
    plt.figure(figsize=fig_size)

    for i in range(nr_exp):
        plt.semilogx(nr_traces_vec, G[i, :],
                     color=slines[i]['Color'],
                     linestyle=slines[i]['LineStyle'],
                     linewidth=slines[i]['LineWidth'],
                     marker=slines[i]['Marker'])

    if 'y' not in options:
        plt.ylim(yrange)

    plt.xlabel(xl_str, fontsize=font_size)
    plt.ylabel('Guessing entropy (bits)', fontsize=font_size)

    if title_ge:
        plt.title(f'Guessing entropy\n{title_ge}', fontsize=font_size)

    if slegend:
        plt.legend(slegend, fontsize=font_small)

    if 'g' in options:
        plt.grid(True)

    if not os.path.exists(rpath):
        os.makedirs(rpath, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"{rpath}{rprefix}guess_entropy.pdf", format='pdf')
    plt.show()
