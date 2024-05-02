import os
import numpy as np
from DET.DET import DET
import argparse

'''
Prerequisites:
    - install matplotlib-3.3.2
'''


def main(args):
    systems = {}

    data_top = np.load(os.path.join(args.top_scores_dir, 'open_set_scores.npz'))
    systems['top'] = {}
    systems['top']['mated'] = data_top['gen']
    systems['top']['non-mated'] = data_top['imp']
    systems['top']['label'] = 'RIE'

    data_bot = np.load(os.path.join(args.bot_scores_dir, 'open_set_scores.npz'))
    systems['bot'] = {}
    systems['bot']['mated'] = data_bot['gen']
    systems['bot']['non-mated'] = data_bot['imp']
    systems['bot']['label'] = 'TE'

    data_comb = np.load(os.path.join(args.combined_scores_dir, 'open_set_scores.npz'))
    systems['combined'] = {}
    systems['combined']['mated'] = data_comb['gen']
    systems['combined']['non-mated'] = data_comb['imp']
    systems['combined']['label'] = 'RIE+TE'


    det = DET(biometric_evaluation_type='identification', plot_title='DET Curves', abbreviate_axes=True,
              plot_eer_line=True)
    det.x_limits = np.array([1e-4, .9])
    det.y_limits = np.array([1e-4, .9])
    det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
    det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

    det.create_figure()

    color = ['blue', 'orange', 'green']

    col = 0
    for system in systems.keys():
        mated = systems[system]['mated']
        non_mated = systems[system]['non-mated']
        det.plot(tar=mated, non=non_mated, label=systems[system]['label'], plot_rocch=True,
                 plot_args=(color[col], '-', '1.5'))
        col += 1

    det.legend_on(loc='upper right')

    plots_dir = os.path.join(args.root_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    det.save(os.path.join(plots_dir, 'DET'), 'png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating DET curves for the top, bot and combined pipelines')
    parser.add_argument('--combined_scores_dir', type=str, help="scores directory for combined pipeline")
    parser.add_argument('--top_scores_dir', type=str, help="scores directory for top pipeline")
    parser.add_argument('--bot_scores_dir', type=str, help="scores directory for bot pipeline")
    parser.add_argument('--root_dir', type=str, help="root dir to create plots folder for")

    args_ = parser.parse_args()
    print(args_)
    main(args_)