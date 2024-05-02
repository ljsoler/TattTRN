import os
from pyeer.cmc_stats import load_scores_from_file, get_cmc_curve, CMCstats
from pyeer.plot import plot_cmc_stats
import argparse


def main(args):
    # combined
    scores_path_comb = os.path.join(args.combined_scores_dir, 'close_set_scores_{}.txt'.format(args.fold))
    tp_path_comb = os.path.join(args.combined_scores_dir, 'close_set_scores_tp_{}.txt'.format(args.fold))
    scores_comb = load_scores_from_file(scores_path_comb, tp_path_comb)
    ranks_comb = get_cmc_curve(scores_comb, args.r)

    # top
    scores_path_top = os.path.join(args.top_scores_dir, 'close_set_scores_{}.txt'.format(args.fold))
    tp_path_top = os.path.join(args.top_scores_dir, 'close_set_scores_tp_{}.txt'.format(args.fold))
    scores_top = load_scores_from_file(scores_path_top, tp_path_top)
    ranks_top = get_cmc_curve(scores_top, args.r)

    # bot
    scores_path_bot = os.path.join(args.bot_scores_dir, 'close_set_scores_{}.txt'.format(args.fold))
    tp_path_bot = os.path.join(args.bot_scores_dir, 'close_set_scores_tp_{}.txt'.format(args.fold))
    scores_bot = load_scores_from_file(scores_path_bot, tp_path_bot)
    ranks_bot = get_cmc_curve(scores_bot, args.r)

    # Stats
    plots_dir = os.path.join(args.root_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Creating stats
    stats = [CMCstats(exp_id='RIE', ranks=ranks_bot),
             CMCstats(exp_id='TE', ranks=ranks_top),
             CMCstats(exp_id='RIE+TE', ranks=ranks_comb)
    ]

    # Plotting
    plot_cmc_stats(stats, args.r, save_path=plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating CMC curves for the top, bot and combined pipelines')
    parser.add_argument('--combined_scores_dir', type=str, help="scores directory for combined pipeline")
    parser.add_argument('--top_scores_dir', type=str, help="scores directory for top pipeline")
    parser.add_argument('--bot_scores_dir', type=str, help="scores directory for bot pipeline")
    parser.add_argument('--root_dir', type=str, help="root dir to create plots folder for")
    parser.add_argument('--r', type=int, help="ranks")
    parser.add_argument('--fold', type=int, help="fold number to plot the results for")


    args_ = parser.parse_args()
    print(args_)
    main(args_)
