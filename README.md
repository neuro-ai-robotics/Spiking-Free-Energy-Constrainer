# Codebase for Spiking Active Inference Controller

### A few files that "build" the simulations and their corresponding visualizations:
- PlantClass has the definitions of the linear dynamical systems and their respective plants
- ControllerClass has the definitions of the controllers used throughout the paper
- SimulatorClass packages the Plant and Controller into a single Simulator class that can be run for results
- DisplayClass has the code for the PyGame animations of the systems (some or all of it may be broken)
- PlotsClass has the definitions for all the plots generated (and the table)

### Furthermore, the Scripts that can be run to actually get the results:
- Script.py produces all the main plots and the table for the paper
- ScriptSupplementary.py produces the supplementary plots for the Appendices
- ScriptAnimation.py produces an animated window showing a live simulation of the system (some or all of it may be broken)
