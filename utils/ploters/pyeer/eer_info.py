import numpy as np

from .eer_stats import calculate_roc, calculate_roc_hist, calculate_roc_auc,\
    get_fmr_op, get_fnmr_op, get_eer_values, Stats, get_decidability_value,\
    get_youden_index, get_matthews_ccoef


def get_eer_stats(gen_scores, imp_scores, hformat=False, ds_scores=False):
    """Calculates EER associated statistics

    Keyword Arguments:
    @param gen_scores: The genuine scores
    @type gen_scores: list
    @param imp_scores: The impostor scores
    @type imp_scores: list
    @param id: An id for the experiment
    @type id: str
    @param hformat: Indicates whether the impostor scores are in histogram
        format
    @type hformat: bool
    @param ds_scores: Indicates whether the input scores are dissimilarity
        scores
    @type ds_scores: bool
    """
    if hformat:
        # Calculating probabilities histogram format
        roc_info = calculate_roc_hist(gen_scores, imp_scores,
                                      ds_scores, rates=False)
        gnumber = float(len(gen_scores))
        inumber = float(sum(imp_scores))
    else:
        # Calculating probabilities using scores as thrs
        roc_info = calculate_roc(gen_scores, imp_scores,
                                 ds_scores, rates=False)
        gnumber = len(gen_scores)
        inumber = len(imp_scores)

    # Unboxing probability rates and info
    thrs, fm, fnm = roc_info
    fmr = fm / inumber
    fnmr = fnm / gnumber

    # Estimating EER
    eer_ind, eer_low, eer_high, eer = get_eer_values(fmr, fnmr)
    eer_th = thrs[eer_ind]

    # Estimating FMR operating points
    ind, fmr0 = get_fmr_op(fmr, fnmr, 0)
    fmr0_th = thrs[ind]

    ind, fmr1000 = get_fmr_op(fmr, fnmr, 0.001)
    fmr1000_th = thrs[ind]

    ind, fmr100 = get_fmr_op(fmr, fnmr, 0.01)
    fmr100_th = thrs[ind]

    ind, fmr20 = get_fmr_op(fmr, fnmr, 0.05)
    fmr20_th = thrs[ind]

    ind, fmr10 = get_fmr_op(fmr, fnmr, 0.1)
    fmr10_th = thrs[ind]

    # Estimating FNMR operating points
    ind, fnmr0 = get_fnmr_op(fmr, fnmr, 0)
    fnmr0_th = thrs[ind]

    # Calculating distributions mean and variance
    gmean = np.mean(gen_scores)
    gstd = np.std(gen_scores)

    if hformat:
        nscores = sum(imp_scores)
        nscores_prob = np.array(imp_scores) / nscores
        scores = np.arange(len(imp_scores))

        imean = (scores * nscores_prob).sum()
        istd = np.sqrt(((scores - imean) ** 2 * nscores_prob).sum())
    else:
        imean = np.mean(imp_scores)
        istd = np.std(imp_scores)

    dec = get_decidability_value(gmean, gstd, imean, istd)

    # Calculating area under the ROC curve
    auc = calculate_roc_auc(fmr, fnmr)

    j_index, j_index_th = get_youden_index(fmr, fnmr)
    j_index_th = thrs[j_index_th]

    mccoef, mccoef_th = get_matthews_ccoef(fm, fnm, gnumber, inumber)
    mccoef_th = thrs[mccoef_th]

    # Stacking stats
    return Stats(thrs=thrs, fmr=fmr, fnmr=fnmr, auc=auc, eer=eer,
                 fmr0=fmr0, fmr100=fmr100, fmr1000=fmr1000,
                 fmr20=fmr20, fmr10=fmr10, fnmr0=fnmr0,
                 gen_scores=gen_scores, imp_scores=imp_scores,
                 gmean=gmean, gstd=gstd, imean=imean, istd=istd,
                 eer_low=eer_low, eer_high=eer_high, decidability=dec,
                 j_index=j_index, j_index_th=j_index_th, eer_th=eer_th,
                 mccoef=mccoef, mccoef_th=mccoef_th, fmr0_th=fmr0_th,
                 fmr1000_th=fmr1000_th, fmr100_th=fmr100_th,
                 fmr20_th=fmr20_th, fmr10_th=fmr10_th, fnmr0_th=fnmr0_th)
