# ElectronChainDutton.jl

`ElectronChainDutton` is a Julia package that simulates the time-evolution of electron transfers using Dutton empirical kinetics. The model computes electron flow across a protein scaffold containing a bound donor, a photo-activated donor in ground or excited states, and a configurable number of acceptor molecules. It also integrates bimolecular electron transfer and binding kinetics involving oxidized and reduced donors, as well as terminal acceptors in the bulk liquid.

## Core Features
* Computes intra-chain (geminate) and inter-chain (bimolecular) electron transfer rates using the Moser and Dutton mathematical formulation for both downhill and uphill reactions.
* Automatically extracts 3D atomic coordinates from CIF files to calculate exact separation distances between ligand molecules.
* Simulates photo-excitation utilizing configurable step, square, or Gaussian light pulses.
* Solves the stiff reaction differential equations using the `Rodas5P` solver to track ensemble-averaged fractional occupancies of all possible chain states over time.

## Dependencies
The package relies on the following standard Julia libraries and external wrappers:
* `DifferentialEquations`
* `Statistics` and `LinearAlgebra`
* `TOML` and `Printf`
* `PyCall` and `PyPlot` (initializes with Qt5 for live rendering or falls back to the Agg backend for headless environments)

## How to Run the Simulation

To execute a simulation, you must provide a TOML configuration file that defines the ligand coordinate sources, thermodynamic parameters, diffusion rates, geometric dimensions, and pulse parameters. 

1. **Initialize Parameters**: Load your physical constants and biological structures from your configuration file.
```Julia
using ElectronChainDutton
p = ElectronChainDutton.build_input_parameters("path/to/your/config.toml")
```

2. **Run the Solver**: Pass the initialized parameters to the solver function, which calculates the chain state indexes and runs the ODE time-evolution.

```Julia
solution, ode_p = ElectronChainDutton.run_simulation(p)
```

3. **Generate Outputs**: Export the data and automatically plot the kinetics.

```Julia
ElectronChainDutton.plot_and_data_output(solution, ode_p, p)
```

## Outputs
Running the data output function creates an output_data_n_figs directory containing the following deliverables:
* results.toml: A data summary file recording the moles of incident photons, total terminal acceptor produced, total efficiency, and overall quantum efficiency.
* Energy Landscapes: A physical map plotting the midpoint voltage versus distance for each molecule, as well as heatmaps of the Dutton kinetic reaction rates.
* State Occupancy Tracking: Time-series figures mapping the temporal fraction of oxidized/reduced bound donors, photo-donors, and acceptors directly against the applied photon pulse profile.
* Flow & Concentration Kinetics: Time-series graphs showing the reaction flow rates and the shifts in molar concentrations of dissolved donors and terminal acceptors in the bulk solution.