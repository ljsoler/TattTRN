import numpy as np
import os
from DET import DET

input_path = '/Users/soler/DATABASES/HAND/HaGRID/output_MKD_gestures'
output_path = '/Users/soler/DATABASES/HAND/HaGRID/plots/MKD_gestures'
hand = 'right'

det = DET(biometric_evaluation_type='algorithm', plot_title='Biometric Performance {} Hand'.format(hand), abbreviate_axes=True, plot_eer_line=True)
det.x_limits = np.array([1e-4, .5])
det.y_limits = np.array([1e-4, .5])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

det.create_figure()

#loading mated and non-mated comparisons
gestures = os.listdir(input_path)
for i in range(len(gestures)//2 + 1):

    if os.path.exists(os.path.join(input_path, gestures[i], hand, 'mated_comparisons.txt')):
        mated = np.loadtxt(os.path.join(input_path, gestures[i], hand, 'mated_comparisons.txt'))
        non_mated = np.loadtxt(os.path.join(input_path, gestures[i], hand, 'non_mated_comparisons.txt'))

        det.plot(tar=mated, non=non_mated, label=gestures[i], plot_rocch=True)

det.legend_on(loc='upper right')
det.save(os.path.join(output_path, 'biometric-performance_{}_1'.format(hand)), 'pdf')


det = DET(biometric_evaluation_type='algorithm', plot_title='Biometric Performance {} Hand'.format(hand), abbreviate_axes=True, plot_eer_line=True)
det.x_limits = np.array([1e-4, .5])
det.y_limits = np.array([1e-4, .5])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

det.create_figure()

#loading mated and non-mated comparisons
for i in range(len(gestures)//2 + 1, len(gestures)):

    if os.path.exists(os.path.join(input_path, gestures[i], hand, 'mated_comparisons.txt')):
        mated = np.loadtxt(os.path.join(input_path, gestures[i], hand, 'mated_comparisons.txt'))
        non_mated = np.loadtxt(os.path.join(input_path, gestures[i], hand, 'non_mated_comparisons.txt'))

        det.plot(tar=mated, non=non_mated, label=gestures[i], plot_rocch=True)

det.legend_on(loc='upper right')
det.save(os.path.join(output_path, 'biometric-performance_{}_2'.format(hand)), 'pdf')


