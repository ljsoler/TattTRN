import numpy as np
import os


tar0 = np.loadtxt('/Volumes/JULIA/FV/FV/RA/9x9_10bit/baseline/mated.txt')
non0 = np.loadtxt('/Volumes/JULIA/FV/FV/RA/9x9_10bit/baseline/attack.txt')

tar1 = np.loadtxt('/Volumes/JULIA/FV/FV/RA/9x9_10bit/adversarial/mated.txt')
non1 = np.loadtxt('/Volumes/JULIA/FV/FV/RA/9x9_10bit/adversarial/attacks.txt')


from DET import DET

det = DET(biometric_evaluation_type='PAD', plot_title='', abbreviate_axes=True, plot_eer_line=True)

det.x_limits = np.array([1e-4, .5])
det.y_limits = np.array([1e-4, .5])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

det.create_figure()
det.plot(tar=tar0, non=non0, label='W/o-Squared Attack', plot_rocch=True)
det.plot(tar=tar1, non=non1, label='W-Squared Attack', plot_args=((0, 0, 1.0), '-', 2))


det.legend_on(loc='upper right')
det.save('/Volumes/JULIA/FV/FV/RA/9x9_10bit/output/DET-FV_squared-attack_RA', 'eps')


