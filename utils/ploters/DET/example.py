import numpy
tar0 = numpy.random.normal(loc=3, scale=2, size=1000)
non0 = numpy.random.normal(loc=0, scale=1, size=5000)
tar1 = 25*numpy.random.beta(a=2,b=.5,size=1000)
non1 = numpy.random.chisquare(df=3,size=5000)
tar2 = numpy.random.beta(a=.9,b=.5,size=1000)
non2 = numpy.random.uniform(low=-4,high=.1,size=5000)

from DET import DET
det = DET(biometric_evaluation_type='algorithm', plot_title='FMR-FNMR')
det.create_figure()
det.plot(tar=tar0, non=non0, label='system 0')
det.plot(tar=tar1, non=non1, label='system 1')
det.plot(tar=tar2, non=non2, label='system 2')
det.legend_on()
det.save('example_algorithm', 'png')


det = DET(biometric_evaluation_type='algorithm', plot_title='FMR-FNMR')

det.x_limits = numpy.array([1e-4, .5])
det.y_limits = numpy.array([1e-4, .5])
det.x_ticks = numpy.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.x_ticklabels = numpy.array(['0.1', '1', '5', '20', '40'])
det.y_ticks = numpy.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
det.y_ticklabels = numpy.array(['0.1', '1', '5', '20', '40'])

det.create_figure()
det.plot(tar=tar0, non=non0, label='system 0')
det.plot(tar=tar1, non=non1, label='system 1')
det.plot(tar=tar2, non=non2, label='system 2')
det.legend_on()
det.save('example_algorithm_axes', 'png')


