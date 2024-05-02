import numpy as np
import os

#Non-tattoos
tar0 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/ABD-Net/det/original/mated_comparisons.txt')
non0 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/ABD-Net/det/original/non_mated_comparisons.txt')

tar1 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/MBA-Net/det/original/mated_comparisons.txt')
non1 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/MBA-Net/det/original/non_mated_comparisons.txt')

tar2 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/RGA-Net/det/original/mated_comparisons.txt')
non2 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/RGA-Net/det/original/non_mated_comparisons.txt')

#Tattoos
tar3 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/ABD-Net/det/tattoo/mated_comparisons.txt')
non3 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/ABD-Net/det/tattoo/non_mated_comparisons.txt')

tar4 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/MBA-Net/det/tattoo/mated_comparisons.txt')
non4 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/MBA-Net/det/tattoo/non_mated_comparisons.txt')

tar5 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/RGA-Net/det/tattoo/mated_comparisons.txt')
non5 = np.loadtxt('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/RGA-Net/det/tattoo/non_mated_comparisons.txt')



from DET import DET

det = DET(biometric_evaluation_type='algorithm', plot_title='', abbreviate_axes=True, plot_eer_line=True,cleanup_segments_distance=0.1)

det.x_limits = np.array([1e-3, .80])
det.y_limits = np.array([1e-3, .80])
det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2, 60e-2, 80e-2])
det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40', '60', '80'])
det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2, 60e-2, 80e-2])
det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40', '60', '80'])

det.create_figure()
det.plot(tar=tar0, non=non0, label='ABD-Net(No tattoo)', plot_rocch=True, plot_args=('darkred', '-', '1.5'))
det.plot(tar=tar1, non=non1, label='MBA-Net(No tattoo)', plot_args=('darkblue', '-', '1.5'))
det.plot(tar=tar2, non=non2, label='RGA-Net(No tattoo)', plot_args=('darkgreen', '-', '1.5'))

det.plot(tar=tar3, non=non3, label='ABD-Net(Tattoo)', plot_args=('darkred', '--', '1.5'))
det.plot(tar=tar4, non=non4, label='MBA-Net(Tattoo)', plot_args=('darkblue', '--', '1.5'))
det.plot(tar=tar5, non=non5, label='RGA-Net(Tattoo)', plot_args=('darkgreen', '--', '1.5'))


# det.legend_on(loc='upper right')
det.legend_on(bbox_to_anchor=(1.55, 1),fontsize="10" ,loc='upper right')
# det.legend_off()
det.save('/Users/soler/DATABASES/HAND/11K-Hands/Split_feat_database/LD/LD_DET', 'pdf')


