import numpy as np
import os


tar0 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADFirst/FV/scores/g.txt')
non0 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADFirst/FV/scores/i.txt')
tar1 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADSecond/FMR10/FV/scores/g.txt')
non1 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADSecond/FMR10/FV/scores/i.txt')

tar2 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADSecond/FMR100/FV/scores/g.txt')
non2 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADSecond/FMR100/FV/scores/i.txt')
tar3 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADSecond/ZeroFMR/FV/scores/g.txt')
non3 = np.loadtxt('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/PADSecond/ZeroFMR/FV/scores/i.txt')


from DET import DET

det = DET(biometric_evaluation_type='PAD', plot_title='', abbreviate_axes=True, plot_eer_line=True)

det.x_limits = np.array([1e-4, .5])
det.y_limits = np.array([1e-4, .5])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

det.create_figure()
det.plot(tar=tar0, non=non0, label='PAD first', plot_rocch=True)
det.plot(tar=tar1, non=non1, label='BR first (FMR=10%)', plot_args=((0, 0, 1.0), 2))
det.plot(tar=tar2, non=non2, label='BR first (FMR=1%)')
det.plot(tar=tar3, non=non3, label='BR first (FMR=0%)')


det.legend_on(loc='upper right')
det.save('/Users/soler/DATABASES/FACE/CSMAD-Mobile/CSMAD-Mobile/output/CSMAD_known_attacks', 'pdf')


