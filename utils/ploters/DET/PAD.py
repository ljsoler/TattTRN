import numpy as np
import os

denseNet = '/Users/soler/Downloads/Aly'

tar0 = np.loadtxt(os.path.join(denseNet, 'mated.txt'))
non0 = np.loadtxt(os.path.join(denseNet, 'non_mated.txt'))

# tar0 = np.loadtxt(os.path.join(denseNet, 'baseline/bm1.txt'))
# non0 = np.loadtxt(os.path.join(denseNet, 'baseline/am2.txt'))
# tar1 = np.loadtxt(os.path.join(denseNet, 'region/bm1.txt'))
# non1 = np.loadtxt(os.path.join(denseNet, 'region/am2.txt'))

# tar2 = np.loadtxt(os.path.join(denseNet, 'genuine_STFT_CQT_Eval.txt'))
# non2 = np.loadtxt(os.path.join(denseNet, 'impostor_STFT_CQT_Eval.txt'))
# tar3 = np.loadtxt(os.path.join(denseNet, 'baseline_eval_gen.txt'))
# non3 = np.loadtxt(os.path.join(denseNet, 'baseline_eval_imp.txt'))

# tar4 = np.loadtxt(os.path.join(denseNet, 'genuine_STFT_CQT_Eval_CD.txt'))
# non4 = np.loadtxt(os.path.join(denseNet, 'impostor_STFT_CQT_Eval_CD.txt'))
# tar5 = np.loadtxt(os.path.join(denseNet, 'genuine_CD.txt'))
# non5 = np.loadtxt(os.path.join(denseNet, 'impostors_CD.txt'))

from DET import DET

det = DET(biometric_evaluation_type='PAD', plot_title='', abbreviate_axes=True, plot_eer_line=True)

det.x_limits = np.array([1e-4, .5])
det.y_limits = np.array([1e-4, .5])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

det.create_figure()
det.plot(tar=tar0, non=non0, label='DeepPixelBis(Full Face)', plot_rocch=True)
# det.plot(tar=tar1, non=non1, label='DeepPixelBis(Central Face)', plot_args=((0, 0, 1.0), '-', 2))
# det.plot(tar=tar2, non=non2, label='Dual-Stream (Unknown-attacks)')
# det.plot(tar=tar3, non=non3, label='FV (Unknown-attacks)')

# det.plot(tar=tar4, non=non4, label='Dual-Stream (Cross-database)')
# det.plot(tar=tar5, non=non5, label='FV (Cross-database)')

det.legend_on(loc='upper right')
det.save('/Users/soler/Downloads/Aly/example', 'pdf')


