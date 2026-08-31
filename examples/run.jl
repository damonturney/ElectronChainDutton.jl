##### Systems simulated previously 
# |  photo-loop paper 2026 Mutter, gaussian pulse     |  pauls 3-molecule system, gaussian pulse          |  Modular bundles connected by loops #1            | 
# |  Fe2+-heme in loop, Fe3+-heme in helix, ZnPPIX    |                                                   |                                                   |
# |    u0[5] = 1.0                                    |                                                   |                                                   |
# |    pulse_param1 = 0.005,    V_D   = 0.3           |    pulse_param1 = 0.005,  V_D   = 0.3,            |    pulse_param1 = 0.005,  V_D   = 0.3,            |       
# |    pulse_param2 = 100.0,    V_hD  = 0.8           |    pulse_param2 = 100.0,  V_hD  = 0.8,            |    pulse_param2 = 100.0,  V_hD  = 0.8,            |       
# |    pulse_param3 = 5,        V_gPD = 1.0           |    pulse_param3 = 5,      V_gPD = 1.0,            |    pulse_param3 = 5,      V_gPD = 1.0,            |       
# |    loc_D = -500.0           V_ePD = V_gPD - 1.8   |    loc_D = -500.0,        V_ePD = V_gPD - 1.8,    |    loc_D = -500.0,        V_ePD = V_gPD - 1.8,    |          
# |    loc_PD = 0.0             V_hPD = V_gPD         |    loc_PD = 0.0,          V_hPD = V_gPD,          |    loc_PD = 0.0,          V_hPD = V_gPD,          |         
# |    loc_A1 = -8              V_A1  = -0.185        |    loc_A1 = -8,           V_A1  = -0.185,         |    loc_A1 = -8,           V_A1  = -0.185,         |         
# |    loc_A2 = 17              V_A2  = -0.350        |    loc_A2 = 17,           V_A2  = -0.350,         |    loc_A2 = 17,           V_A2  = -0.350,         |       
# |    loc_At = loc_A2 + 500    V_At  = 0.3           |    loc_At = loc_A2 + 500, V_At  = 0.3,            |    loc_At = loc_A2 + 500, V_At  = 0.3,            | 





# Comment out the line below if you don't want it re-compiled
include("../src/ElectronChainDutton.jl")

# Generate the strongly-typed parameters locally
p = ElectronChainDutton.build_input_parameters("config.toml");

# Pass them to the precompiled module
solution, ode_p = ElectronChainDutton.run_simulation(p);
ElectronChainDutton.plot_and_data_dump(solution, ode_p, p)