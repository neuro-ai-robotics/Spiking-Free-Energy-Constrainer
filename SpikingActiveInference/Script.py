#Imports
from SimulatorClass import * 
from PlotsClass import Plots

#Make a plots object
plots = Plots()

# Set the simulation time
T = 30

# Define the linear systems to be used in the simulation
linear_systems = ['SMD', 'coupledSMD', '2D_masses']

# Create an instance of the simulator with the specified parameters for plotting
# Make Fig 2 - SCN controlling SMD
simulatorFig2 = Simulator_Basic(T, system_type = linear_systems[0], controller_type = 'SCN')
vals_Fig2 = simulatorFig2.run()
plots.make_Fig2(vals_Fig2, title='PaperFigs/Fig2.png')
print('Fig2 done')

# Make Fig 3 - SCN controlling coupled SMD
simulatorFig3 = Simulator_Basic(T, system_type = linear_systems[1], controller_type = 'SCN')
vals_Fig3 = simulatorFig3.run()
plots.make_Fig2(vals_Fig3, title='PaperFigs/Fig3.png')
print('Fig3 done')

# Make Fig 4 - SCN controlling 2D masses
simulatorFig4 = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'SCN')
vals_Fig4 = simulatorFig4.run()
plots.make_Fig2(vals_Fig4, title='PaperFigs/Fig4.png')
print('Fig4 done')

# Make Fig5 - Robustness to External Perturbations - 2D masses
simulatorFig5 = Simulator_Compare_External(T)
vals_Fig5 = simulatorFig5.run()
plots.make_Fig5(vals_Fig5, title='PaperFigs/Fig5.png')
print('Fig5 done')

# Make Fig6 - Robustness to Internal Perturbations - 2D masses
simulatorFig6 = Simulator_Compare_Internal(T)
vals_Fig6 = simulatorFig6.run()
plots.make_Fig5(vals_Fig6, title='PaperFigs/Fig6.png')
print('Fig6 done')

# Make Fig 8 - Different Dynamics
simulatorFig8 = Simulator_Basic(T, system_type = '2D_masses_different', controller_type = 'SCN')
vals_Fig8 = simulatorFig8.run()
plots.make_Fig8(vals_Fig8, title='PaperFigs/Fig8.png')
print('Fig8 done')

# Make Fig 9 - Make a heatmap of the noise robustness
simulatorHeatMap = Simulator_Compare_Noise(T)
MSE_heatmap, noise_ctrl, noise_obs = simulatorHeatMap.run_compare()
plots.make_Fig9(MSE_heatmap, noise_ctrl, noise_obs, title='PaperFigs/Fig9.png')
print('Fig9 done')

#make Fig 10 - Energy Comparison in equal conditions between Nengo, SCN and Gradient SCN
simulatorNengo_Fig10 = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'ActInf_Nengo')
vals_Fig10_Nengo = simulatorNengo_Fig10.run()

simulatorGradient_Fig10 = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'ActInf_SCN')
vals_Fig10_Gradient = simulatorGradient_Fig10.run()

simulatorSFEC_Fig10 = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'SCN')
vals_Fig10_SFEC = simulatorSFEC_Fig10.run()

plots.make_Fig10(vals_Fig10_Nengo, vals_Fig10_Gradient, vals_Fig10_SFEC, title='PaperFigs/Fig10.png')
print('Fig10 done')


averages = np.zeros((9, 10))
for i in range(10):
    simulatorNengo = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'ActInf_Nengo')
    vals_Nengo = simulatorNengo.run()

    simulatorGradient = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'ActInf_SCN')
    vals_Fig10_Gradient = simulatorGradient.run()
    #plots.make_Fig2(vals_Fig10_Gradient, title=f'Gradient_Fig10_run{i+1}.png')

    simulatorSFEC = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'SCN')
    vals_Fig10_SFEC = simulatorSFEC.run()

    averages[:, i] += plots.make_metrics(vals_Fig10_Nengo, vals_Fig10_Gradient, vals_Fig10_SFEC, time=simulatorNengo_Fig10.time)

plots.make_table(averages)
print('Table 1 done')