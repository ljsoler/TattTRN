import numpy as np
import os


tar0 = np.loadtxt('/Users/soler/DATABASES/MORPH/HDA-FaceManipulation-Embeddings/arcface/features/frgc/output/mated_comparisons.txt')
non0 = np.loadtxt('/Users/soler/DATABASES/MORPH/HDA-FaceManipulation-Embeddings/arcface/features/frgc/output/non_mated_comparisons.txt')

tar1 = np.loadtxt('/Users/soler/DATABASES/MORPH/HDA-FaceManipulation-Embeddings/trained/arcface/frgc/output/mated_comparisons.txt')
non1 = np.loadtxt('/Users/soler/DATABASES/MORPH/HDA-FaceManipulation-Embeddings/trained/arcface/frgc/output/non_mated_comparisons.txt')

from DET import DET

det = DET(biometric_evaluation_type='None', plot_title='', abbreviate_axes=True, plot_eer_line=True)

det.x_label = 'FMR (in %)'
det.y_label = 'FNMR (in %)'

det.x_limits = np.array([1e-4, .5])
det.y_limits = np.array([1e-4, .5])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

det.create_figure()
det.plot(tar=tar0, non=non0, label='ArcFace', plot_rocch=True)
det.plot(tar=tar1, non=non1, label='UTM', plot_rocch=True)


det.legend_on(loc='upper right')
det.save('/Users/soler/DATABASES/MORPH/HDA-FaceManipulation-Embeddings/arcface/features/frgc/output/DET_frgc', 'png')


