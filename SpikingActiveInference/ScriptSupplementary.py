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
# Make SupFig 2 - Nengo controlling SMD
simulatorFig2 = Simulator_Basic(T, system_type = linear_systems[0], controller_type = 'ActInf_Nengo')
vals_Fig2_Nengo = simulatorFig2.run()
plots.make_Fig2(vals_Fig2_Nengo, title='SupplementaryFigs/SupFig2_Nengo.png')
print('SupFig2_Nengo done')

simulatorFig2 = Simulator_Basic(T, system_type = linear_systems[0], controller_type = 'ActInf_SCN')
vals_Fig2_Gradient = simulatorFig2.run()
plots.make_Fig2(vals_Fig2_Gradient, title='SupplementaryFigs/SupFig2_Gradient.png')
print('SupFig2_Gradient done')

# Make SupFig 3 - Nengo controlling coupled SMD
simulatorFig3 = Simulator_Basic(T, system_type = linear_systems[1], controller_type = 'ActInf_Nengo')
vals_Fig3_Nengo = simulatorFig3.run()
plots.make_Fig2(vals_Fig3_Nengo, title='SupplementaryFigs/SupFig3_Nengo.png')
print('SupFig3_Nengo done')

simulatorFig3 = Simulator_Basic(T, system_type = linear_systems[1], controller_type = 'ActInf_SCN')
vals_Fig3_Gradient = simulatorFig3.run()
plots.make_Fig2(vals_Fig3_Gradient, title='SupplementaryFigs/SupFig3_Gradient.png')
print('SupFig3_Gradient done')

# Make SupFig 4 - Nengo controlling 2D masses
simulatorFig4 = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'ActInf_Nengo')
vals_Fig4_Nengo = simulatorFig4.run()
plots.make_Fig2(vals_Fig4_Nengo, title='SupplementaryFigs/SupFig4_Nengo.png')
print('SupFig4_Nengo done')

simulatorFig4 = Simulator_Basic(T, system_type = linear_systems[2], controller_type = 'ActInf_SCN')
vals_Fig4_Gradient = simulatorFig4.run()
plots.make_Fig2(vals_Fig4_Gradient, title='SupplementaryFigs/SupFig4_Gradient.png')
print('SupFig4_Gradient done')
