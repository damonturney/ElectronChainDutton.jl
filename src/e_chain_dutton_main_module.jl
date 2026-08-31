#=                                                             
Created 20260120
@author: dturney

V_eEPD = electron energy (volts) of the excited photo-absorber
V_eGPD = electron energy (volts) of the ground state photo-activated-donor (PD)
V_eA1  = electron energy (volts) of Acceptor 1
V_eA2  = electron energy (volts) of Acceptor 2

             <- dist_PD_eA1 ->  |  <----- dist_eA1_eA2 ------> |                        
        Vm_eEPD       
                                                            | dV_eEPD_eA1 = difference in voltage from excited photo-absorber to Acceptor 1
                                                            | dV_eA1_eGPD = difference in voltage from molecule 1 to ground state photo-activated-donor (D)
                                V_eA1                        |
                                                            | dV_eA1_eA2 =  difference in voltage from molecule 1 to Acceptor 2
                                                       V_eA2 |
                                                            | dV_eA2_eGPD = difference in voltage from molecule 2 to ground state photo-activated-donor (PD)
                                                            |
D                                                           |
                                                            |
                                                            |
                                                            |
        V_eGPD                                               |
                                                            |

Naming of states of the sites on a single scaffold:
BD         a bound donor molecule in the binding site of a single protein scaffold, can be 0 or 1
eBD        an electron orbital of a single bound donor molecule, can be 0 or 1 or 2 (if no ligand is bound, aka binding site is empty)
eGPD        an electron orbital of the ground state photo-excitable-donor (PD), can be 0 or 1
eEPD        an electron orbital of the excited state photo-excitable-donor (PD), can be 0 or 1
eA1         an electron orbital of the Acceptor molecule 1, can be 0 or 1
eA2         an electron orbital of the Acceptor molecule 2, can be 0 or 1

Naming of the ensemble average occupancy (for a collection of many scaffolds)
ens_ave_BD       average fraction of the binding sites that are occupied by a Donor molecule,  can be any number between 0 and 1
ens_ave_eBD      average fraction of the electron orbitals occupied on the bound donor molecules,  can be any number between 0 and 1
ens_ave_eGPD      average fraction of the electron orbitals occupied on the ground state photo-excitable-donor (PD) molecules,  can be any number between 0 and 1
ens_ave_eEPD      average fraction of the electron orbitals occupied on the excited state photo-excitable-donor (PD) molecules,  can be any number between 0 and 1
ens_ave_eA1       average fraction of the electron orbitals occupied on the Acceptor molecule 1,  can be any number between 0 and 1
ens_ave_eA2       average fraction of the electron orbitals occupied on the Acceptor molecule 2,  can be any number between 0 and 1


To Do:
Does the binding of D require Smoluchowski and Eigen-Fuoss equations?  Currently I don't use them for this reaction.
Add bimolecular reaction between oxD and reD and the chain ligands.
double check that your bimolecular reaction flux is correct: flow = k_obs * u[rxn_reactant_A_cs_idx] * u[rxn_reactant_B_cs_idx] * ode_p.p_in.conc_chains   

=# 

module e_chain_dutton_main_module
using Printf
using DifferentialEquations
using Statistics:mean
using Statistics:mean
using LinearAlgebra:norm

using Printf

import PyCall
PyCall.pygui(:qt5)
import PyPlot as plt
plt.close("all")
plt.ioff()






function chain_state_binary_to_idx_rep(cs)
    # eBD = cs[1], eGPD = cs[2],   eEPD = cs[3],    eA1 = cs[4], .... eAn = cs[num_cs_bits]
    # reverse order: eA1 = cs[4], eEPD = cs[3], eGPD = cs[2], eBD = cs[1]]
    num_cs_bits = length(cs)                          # eBD IS ACTUALLY A "TRIT" NOT A BIT BECAUSE --> eBD = 0 means occupied by oxBD,  eBD = 1 means occupied by reBD,   eBD = 2 means empty binding site
    n = num_cs_bits - 3                               #  n is the number of Acceptors.
    An_sum = 0
    @inbounds for i in 1:(num_cs_bits-3)              # @inbounds tells the Julia compiler to skip safety checks during the loop, saving CPU cycles on a loop that is guaranteed to remain within the vector's lengt
        An_sum += cs[num_cs_bits - i + 1] << (i - 1)  # Reverse the order of the elements in eAn. This way you can enter eAn = [0,1] to say A1 = 0, A2 = 1.  Otherwise it's counterintuitive.
    end
    #        eAn = cs[4...],      eEPD = cs[3],         eGPD = cs[2],         eBD = cs[1],       
    return      An_sum      +   (cs[3] << (n))    + (cs[2] << (n+1))  + (cs[1] * (1 << (n+2)))  + 1  # << is bit shift (equivalent to 2^) but faster.  +1 bc we want the first state to be 1, not 0.   
end


function chain_state_idx_to_binary_rep(n::Int, idx::Int)    #  n is the number of Acceptors.
    eBD = trunc(Int,(idx-1)/(1 << (n+2)))                   # eBD = 0 means occupied by oxBD,  eBD = 1 means occupied by reBD,   eBD = 2 means empty binding site
    temp = idx - eBD * (1 << (n+2)) - 1                     # -1 bc binary starts at 0, but u[i] idx starts at 1
    eGPD = (temp >> (n+1)) & 1     # Extract states using fast bitwise shifts and masks
    eEPD = (temp >> (n+0)) & 1     # Extract states using fast bitwise shifts and masks
    eAn = Vector{Int}(undef, n)    # eAn requires special treatment 
    @inbounds for i in 1:n          # @inbounds tells the Julia compiler to skip safety checks during the loop, saving CPU cycles on a loop that is guaranteed to remain within the vector's length
        eAn[i] = (temp >> (n - i)) & 1
    end
    return (eBD, eGPD, eEPD, eAn...)
end


function gaussian_pulse(t, amp, center, width)
    # Clamp the exponential to avoid underflow/denormalized numbers
    val = -((t - center)^2) / (2 * width^2)
    return amp * exp(val)
end

function k_dutton_downhill(dG, distance, lambda_dG)  # Feed this function only negative values of dG
    # k = 10^( 13.0 - 0.6*(distance - 3.6) - 3.1*(dG + lambda_dG)**2.0 /lambda_dG )         # form given in Moser & Dutton 2010
    # k = 10^( 15.16 - 0.6*distance - 3.1*(dG + lambda_dG)^2.0 /lambda_dG )                 # Moser and Dutton were incabable of algebraic simplification?
    return 10^(15.16 - 0.6*distance - 3.1*((dG + lambda_dG)^2.0)/lambda_dG)                 # returns rate in 1/seconds
end

function k_dutton_uphill(dG, distance, lambda_dG)  # Feed this function only positive values of dG                                
    # k = 10^( 13.0 - 0.6*(distance - 3.6) - 3.1*(-dG + lambda_dG)**2.0/lambda_dG - dG/0.06)        # form given in Moser & Dutton 2010
    # k = 10^( 15.16 - 0.6*distance - 3.1*(-dG + lambda_dG)^2.0 /lambda_dG - dG/0.06 )              # Moser and Dutton were incabable of algebraic simplification?
    return 10^( 15.16 - 0.6*distance - 3.1*(-dG + lambda_dG)^2.0/lambda_dG - dG/0.06 )              # returns rate in 1/seconds
end

function k_dutton_calc(dG, distance, lambda_dG)                                
    if dG >=0
        return k_dutton_uphill(dG, distance, lambda_dG)
    else
        return k_dutton_downhill(dG, distance, lambda_dG)
    end
end




function binary_numbers_w_1bit_frozen(num_bit_positions::Int, bit_pos::Int, value::Int)
    # Bit positions are defined for a binary number like 0101 like below.     BUT eBD ISN'T BINARY!!  eBD = 0 means occupied by oxBD,  eBD = 1 means occupied by reBD,   eBD = 2 means empty binding site
    #  4th  3rd  2nd  1st         which translates to   
    #  0    1    0    1            0*8  +  1*4   + 0*2  + 1*1 = 5
    num_bits_below_bp = bit_pos - 1
    num_bits_above_bp = num_bit_positions - bit_pos
    
    # Pre-allocate the result array for maximum speed
    num_combinations = 1 << (num_bit_positions - 1)
    result = Vector{Int}(undef, num_combinations)
    m = 1

    for i in 0:(2^num_bits_above_bp - 1)
        for j in 0:(2^num_bits_below_bp - 1)
            result[m] = i << bit_pos  + value << (bit_pos - 1)  +  j << 0  # the a << b operator converts a to binary representation then shifts a binary bits to the left (padding its RHS with bit zeros), thus 1<<0 = decimal 1 , 1<<2 = decimal 4
            m += 1
        end
    end

    return result     # if you calculate bitstring.(result) on the output from this function you'll see the binary representation of the decimal numbers
end



function chain_states_w_1state_frozen(num_chain_state_bits::Int, bit_pos::Int, frozen_value::Int)
    #  chain orbitals bit positions are defined as:          eBD, eEPD, dGPD, A1,  A1, ... An-1, An
    #  chain_state_bits are defined as:                      eBD, eEPD, dGPD, A1,  A1, ... An-1, An         
    #  n is the number of Acceptors,    the "frozen state" is which bit position in the binary representation ([eBD, eEPD, dGPD, A1,  A1, ... An-1, An] ) is frozen, and frozen_value is the value of that bit.
    #                                                                                            bit positions: n+3,  n+2,  n+1,  n, n-1, ...    2,  1
    n = num_chain_state_bits - 3            # n is the number of Acceptors
    if bit_pos == num_chain_state_bits      # If the bit position is the eBD binding state (bit position n+3), then we treat things differently
        return frozen_value * 2^(n+2) .+ [i for i in 1:(2^(n+2))] 
    else
        set1 = binary_numbers_w_1bit_frozen(n+2, bit_pos, frozen_value) .+ 1
        set2 = set1 .+ 2^(n+2)
        set3 = set2 .+ 2^(n+2) 
        return [set1 ; set2 ; set3]
    end
end



function binary_numbers_w_2bits_frozen(num_bit_positions::Int, bit_pos1::Int, bit_pos1_value::Int, bit_pos2::Int, bit_pos2_value::Int)
    # Bit positions are defined for a binary number like 0101 like this:
    #  4th  3rd  2nd  1st         which translates to   
    #  0    1    0    1                                   0*8  +  1*4   + 0*2  + 1*1 = 5
    
    # Determine which bit position is lower
    if bit_pos1 < bit_pos2
        bp_low,  bp_low_value  = bit_pos1, bit_pos1_value
        bp_high, bp_high_value = bit_pos2, bit_pos2_value
    elseif bit_pos1 > bit_pos2
        bp_low,  bp_low_value  = bit_pos2, bit_pos2_value
        bp_high, bp_high_value = bit_pos1, bit_pos1_value
    elseif bit_pos1 == bit_pos2
        return 0
    end

    num_bits_below_bp_low     = bp_low - 1
    num_bits_btwn_bp_high_low = bp_high - bp_low - 1
    num_bits_above_bp_high    = num_bit_positions - bp_high
    
    # Pre-allocate the result array for maximum speed
    num_combinations = 1 << (num_bit_positions - 2)
    result = Vector{Int}(undef, num_combinations)
    m = 1

    for i in 0:(2^num_bits_above_bp_high - 1)
        for j in 0:(2^num_bits_btwn_bp_high_low) - 1
            for k in 0:(2^num_bits_below_bp_low) - 1
                result[m] = i << bp_high  +  bp_high_value << (bp_high - 1)  +  j << bp_low  +  bp_low_value << (bp_low - 1)  +  k << 0  # the a << b operation converts a to binary representation then shifts a binary bits to the left (padding its RHS with bit zeros), thus 1<<0 = decimal 1 , 1<<2 = decimal 4
                m += 1
            end
        end
    end

    return result     # if you calculate bitstring.(result) on the output from this function you'll see the binary representation of the decimal numbers
end




function chain_states_w_2states_frozen(num_chain_state_bits::Int, bit_pos1::Int, frozen_value1::Int, bit_pos2::Int, frozen_value2::Int)
    n = num_chain_state_bits - 4
    #  num_chain_state_bits is defined as:  eBD, eEPD, dGPD, A1,  A1, ... An-1, An         BUT eBD ISN'T BINARY!!  eBD = 0 means occupied by oxBD,  eBD = 1 means occupied by reBD,   eBD = 2 means empty binding site
    #  n is the number of Acceptors,    frozen_state is which bit position in the binary representation ([ eBD, eEPD, dGPD, A1,  A1, ... An-1, An] ) is frozen, and frozen_value is the value of that bit.
    #                                                                                      bit positions:  n+3,  n+2,  n+1,  n, n-1, ...    2,  1
    # Determine which bit position is lower
    if bit_pos1 < bit_pos2
        bp_low,  bp_low  = bit_pos1, frozen_value1
        bp_high, bp_high = bit_pos2, frozen_value2
    elseif bit_pos1 > bit_pos2
        bp_low,  bp_low_value  = bit_pos2, frozen_value2
        bp_high, bp_high_value = bit_pos1, frozen_value1
    elseif bit_pos1 == bit_pos2
        return 0
    end

    n = num_chain_state_bits - 3            # n is the number of Acceptors

    if bp_high == num_chain_state_bits    # If the bit position is the eBD binding state (bit position n+4), then we treat things differently
        return frozen_value1 * 2^(n+2) .+ binary_numbers_w_1bit_frozen(n+2, bp_low, bp_low_value) .+ 1 
    else
        set1 = binary_numbers_w_2bits_frozen(n+2, bit_pos1, frozen_value1, bit_pos2, frozen_value2) .+ 1
        set2 = set1 .+ 2^(n+2)
        set3 = set2 .+ 2^(n+2)
        return [set1 ; set2 ; set3]
    end
end




# ==========================================
# CIF File: Calculate Ligand Locations and Separation Distances
# ==========================================
function extract_ligands_atomic_coords(cif_file, target_attributes_labels)
    all_ligands_coords = Vector{Vector{Vector{Float64}}}()
    for i in 1:length(target_attributes_labels)
        ligand_coords = Vector{Vector{Float64}}()
        cif_attribute_match_idxs = Vector{Int}() 
        target_attribute_match_idxs = Vector{Int}() 
        column_idx = 1
        x_coord_idx = 0; y_coord_idx = 0; z_coord_idx = 0
        file = open(cif_file, "r") 
        try
            for line in eachline(file)
                stripped_line = strip(line)            #removed spaces in front and back
                split_line = split(stripped_line)
                # A new data block starts with a "#"
                if startswith(stripped_line, "#")
                    cif_attribute_match_idxs = Vector{Int}() 
                    target_attribute_match_idxs = Vector{Int}() 
                    column_idx = 1
                    x_coord_idx = 0; y_coord_idx = 0; z_coord_idx = 0
                elseif startswith(stripped_line, "_atom_site.")
                    if any(stripped_line .== target_attributes_labels[i])
                        push!(cif_attribute_match_idxs, column_idx)                                                # the attribute index in the CIF file's list of all attributes (this line) will be matched up with the index of the same attribute in the user's list of attributes to use (line below)
                        push!(target_attribute_match_idxs, findfirst(==(stripped_line), target_attributes_labels[i]))     # the attribute index in the CIF file's list of all attributes (above line) will be matched up with the index of the same attribute in the user's list of attributes to use (line here))
                    end
                    if contains(stripped_line, "Cartn_x") x_coord_idx = column_idx end
                    if contains(stripped_line, "Cartn_y") y_coord_idx = column_idx end
                    if contains(stripped_line, "Cartn_z") z_coord_idx = column_idx end
                    column_idx += 1
                elseif (startswith(stripped_line, "HETATM") || startswith(stripped_line, "ATOM")) && column_idx > 6 && x_coord_idx > 0 && y_coord_idx > 0 && z_coord_idx > 0 && split_line[cif_attribute_match_idxs] == target_attributes_labels[target_attribute_match_idxs .+ 1]      # Extract coordinates for matching ATOM/HETATM entries
                    push!(ligand_coords, [parse(Float64, split_line[x_coord_idx]), parse(Float64, split_line[y_coord_idx]), parse(Float64, split_line[z_coord_idx])])
                end
            end
        finally
            close(file)
            push!(all_ligands_coords, ligand_coords)
        end
    end
    return all_ligands_coords
end

function get_geometric_center(coords)
    return sum(coords) / length(coords)
end

function get_closest_distance(coords1, coords2)
    min_dist = Inf
    for c1 in coords1
        for c2 in coords2
            d = norm(c1 - c2)
            if d < min_dist
                min_dist = d
            end
        end
    end
    return min_dist
end




function calculate_flows(u, ode_p, p, t)

    # Calculate pulse intensity and eGPD excitation kinetic coefficient
    if p.pulse_type == 0
    elseif p.pulse_type == 1; if t>= p.pulse_param2; k_exc_vs_t = p.pulse_param1; end                           # Step Pulse
    elseif p.pulse_type == 2; if t >= p.pulse_param2 && t <= p.pulse_param3; k_exc_vs_t = p.pulse_param1; end   # Square Pulse
    elseif p.pulse_type == 3; k_exc_vs_t = gaussian_pulse(t, p.pulse_param1, p.pulse_param2, p.pulse_param3)    # Gaussian Pulse
    end
    
    # e- flows for Intra-Chain (aka Geminate) e- transfers. 1st Order Kinetics
    # It's a matrix M_ij where elements in upper-right triangle are the flow from chain orbital i to j: aka outflows from i  
    #                      and elements ni bottom-left triangle are the flow from chain orbital j to i: aka inflows to i 
    flows_matrix_e_gem = Matrix{Float64}(zeros(ode_p.num_orbitals - 1, ode_p.num_orbitals - 1))  # -1 bc we don't consider the eAt orbital bc it's not "geminate" aka bound on the chain
    for i in 1:ode_p.num_orbitals1 - 1
        for j in 1:ode_p.num_orbitals - 1                        # sum of all u[i] for chain states that flow from from chain site i to chain site j
            flows_matrix_e_gem[i,j] = ode_p.k_matrix_e_gem[i, j] * sum(u[chain_states_w_2states_frozen(ode_p.num_orbitals, i, Int(j>i), j, Int(j>i))])  # UNITS: 1/picosecond      The upper-right triangle are the flow from chain orbital i to j: aka outflows from i  .  The elements ni bottom-left triangle are the flow from chain orbital j to i: aka inflows to i .
        end
    end
    flows_matrix_e_gem[2,3] += k_exc_vs_t * sum(u[chain_states_w_2states_frozen(ode_p.num_orbitals, 2, 1, 3, 0)])    # Add the excitation of electrons due to the light pulse


    # e- flows Inter-Chain (aka Bimolecular). 2nd Order Kinetics
    # this matrix is the same as above except has an extra 2 dimensions for flow to liquid state D or At, and uses 2nd order reaction kinetics
    # It's a matrix M_ij where elements in upper-right triangle are the flow from chain orbital i to j: aka outflows from i  
    #                      and elements ni bottom-left triangle are the flow from chain orbital j to i: aka inflows to i 
    flows_matrix_e_bim = Matrix{Float64}(zeros(ode_p.num_orbitals + 1, ode_p.num_orbitals + 1))
    for i in 1:ode_p.num_orbitals + 1
        for j in 1:ode_p.num_orbitals + 1                        # sum of all u[i] for chain states that flow from from chain site i to chain site j
            flows_matrix_e_bim[i,j] = ode_p.k_matrix_e_bim[i, j] * sum(u[chain_states_w_1state_frozen(ode_p.num_orbitals, i, 1)] .* u[chain_states_w_1state_frozen(ode_p.num_orbitals, j, 0)]) # UNITS: 1/picosecond   
        end
    end

    # flows of moles of the reD and oxD and reAt and oxAt molecules due to bimolecular reactions, and flow of the reBD and oxBD binding fractions due to binding exchange
    flow_reD_bind   = ode_p.k_on  * u[ode_p.reD_conc_idx] * sum(u[chain_states_w_1state_frozen(ode_p.num_chain_state_bits, ode_p.num_chain_state_bits, 0)])
    flow_oxD_bind   = ode_p.k_on  * u[ode_p.oxD_conc_idx] * sum(u[chain_states_w_1state_frozen(ode_p.num_chain_state_bits, ode_p.num_chain_state_bits, 0)])
    flow_reD_debind = ode_p.k_off * u[ode_p.reD_conc_idx] * sum(u[chain_states_w_1state_frozen(ode_p.num_chain_state_bits, ode_p.num_chain_state_bits, 1)])
    flow_oxD_debind = ode_p.k_off * u[ode_p.oxD_conc_idx] * sum(u[chain_states_w_1state_frozen(ode_p.num_chain_state_bits, ode_p.num_chain_state_bits, 1)])
    return flows_matrix_e_gem, flows_matrix_e_bim
end









# ==========================================
# 2. Kinetics for ODE Solver , u is a vector elements representing ensemble-averaged occupancy of each chain state (and in the case of u[49] or u[51] a concentration)
# ==========================================
function dudt!(dudt, u, ode_p, t)               # the exclamation mark makes the function modify (mutate) its argum    
    fill!(dudt, 0.0)                           # this fill command replaces all contents of dudt with zeros                                             

    # Calculate pulse intensity (k_exc_vs_t)
    if ode_p.p_in.pulse_type == 1 
        if t >= ode_p.p_in.pulse_param2; k_exc_vs_t = ode_p.p_in.pulse_param1; end
    elseif ode_p.p_in.pulse_type == 2 
        if t >= ode_p.p_in.pulse_param2 && t <= ode_p.p_in.pulse_param3; k_exc_vs_t = ode_p.p_in.pulse_param1; end
    elseif ode_p.p_in.pulse_type == 3 
        photon_flux = gaussian_pulse(t, ode_p.p_in.pulse_param1, ode_p.p_in.pulse_param2, ode_p.p_in.pulse_param3)
    end


    # Photo-excitation dudt
    for (u_idxs_reactant, u_idxs_rxn_product, eGPD_cross_section) in ode_p.idxs_k_const_excite_eGPD
        flow = photon_flux * eGPD_cross_section * 1E-12 * u[u_idxs_reactant]
        dudt[u_idxs_reactant]    -= flow
        dudt[u_idxs_rxn_product] += flow
    end

    # Geminate (intra-chain) dudt
    for (u_idxs_reactant, u_idxs_rxn_product, rxn_k_constant) in ode_p.idxs_k_const_rxns_gem
        flow = rxn_k_constant * u[u_idxs_reactant]
        dudt[u_idxs_reactant]    -= flow
        dudt[u_idxs_rxn_product] += flow
    end


    # Bimolecular / Binding dudt
    reD_conc  = u[ode_p.reD_conc_idx]  # (Extract bulk concentrations)
    oxD_conc  = u[ode_p.oxD_conc_idx]
    oxAt_conc = u[ode_p.oxAt_conc_idx]
    
    for (u_idxs_reactant, u_idxs_rxn_product, k_on) in ode_p.idxs_k_const_bind_oxD    # Binding of oxD onto the chain
        flow = k_on * oxD_conc * u[u_idxs_reactant]
        dudt[u_idxs_reactant]    -= flow
        dudt[u_idxs_rxn_product] += flow
        dudt[ode_p.oxD_conc_idx] -= flow * ode_p.p_in.conc_chains   # 
    end
    
    for (u_idxs_reactant, u_idxs_rxn_product, k_on) in ode_p.idxs_k_const_bind_reD     # Binding of oxD onto the chain
        flow = k_on * reD_conc * u[u_idxs_reactant]
        dudt[u_idxs_reactant]    -= flow
        dudt[u_idxs_rxn_product] += flow
        dudt[ode_p.reD_conc_idx] -= flow * ode_p.p_in.conc_chains   # 
    end

    for (u_idxs_reactant, u_idxs_rxn_product, k_off) in ode_p.idxs_k_const_debind_reD
        flow = k_off * u[u_idxs_reactant]
        dudt[u_idxs_reactant]    -= flow
        dudt[u_idxs_rxn_product] += flow
        dudt[ode_p.reD_conc_idx] += flow * ode_p.p_in.conc_chains   # 
    end

    for (u_idxs_reactant, u_idxs_rxn_product, k_off) in ode_p.idxs_k_const_debind_oxD
        flow = k_off * u[u_idxs_reactant]
        dudt[u_idxs_reactant]    -= flow
        dudt[u_idxs_rxn_product] += flow
        dudt[ode_p.oxD_conc_idx] += flow * ode_p.p_in.conc_chains   # 
    end

    # Electron transfer onto the terminal Acceptor At
    for (u_idxs_reactant, u_idxs_rxn_product, k_dutton) in ode_p.idxs_k_const_bimol_At
        k_obs = 1/(1/ode_p.p_in.k_on_At_bim + 1/(ode_p.p_in.K_d_At_bim * k_dutton))     # Uses a reaction kinetic Smoluchowski k_on and Eigen-Fuoss reaction association constant K_A
        flow = k_obs * oxAt_conc * u[u_idxs_reactant]
        dudt[u_idxs_reactant]     -= flow
        dudt[u_idxs_rxn_product]  += flow
        dudt[ode_p.oxAt_conc_idx] -= flow * ode_p.p_in.conc_chains  # 
        dudt[ode_p.reAt_conc_idx] += flow * ode_p.p_in.conc_chains  # 
    end

    # Inter-Chain Electron Transfer 
    for (rxn_reactant_A_cs_idx, rxn_reactant_B_cs_idx, rxn_product_A_cs_idx, rxn_product_B_cs_idx, k_dutton) in ode_p.precomputed_interchain_ET
        k_obs = 1/(1/ode_p.p_in.k_on_chains_bim + 1/(ode_p.p_in.K_d_chains_bim * k_dutton))                 # Uses a reaction kinetic Smoluchowski k_on and Eigen-Fuoss reaction association constant K_A
        flow = k_obs * u[rxn_reactant_A_cs_idx] * u[rxn_reactant_B_cs_idx] * ode_p.p_in.conc_chains    # Calculate the fractional flow (Rate * u1 * u2 * Total Chain Conc)
        dudt[rxn_reactant_A_cs_idx] -= flow  # reactant A
        dudt[rxn_product_A_cs_idx ] += flow  # product  A
        dudt[rxn_reactant_B_cs_idx] -= flow  # reactant B
        dudt[rxn_product_B_cs_idx ] += flow  # product  B
    end

end




############
### Run Simulation
function run_simulation(p)
    ### LEGEND FOR LIGANDS
    #     Ligands:        BD       PD      A1    A2    A3  ... An                    num_ligands = n + 2       n is the number of acceptor ligands: A1 A2 A3 ... An
    #     ligand site     1        2       3     4     5       n+2
    #
    ### LEGEND FOR ELECTRON ORBITALS                            
    # orbital site        1     2     3    4     5     6        n+2    n+3    n+4    num_orbitals = n + 4               
    # All e- Orbitals:   eBD, eGPD, eEPD, eA1,  eA2,  eA3  ... eAn-1   eAn    eAt        
    # Chain e- orbitals: eBD, eGPD, eEPD, eA1,  eA2,  eA3  ... eAn-1
    # liquid e- orbitals: eD                                                   eAt
    #
    ### LEGEND FOR GEMINATE ELECTRON ORBITALS                            
    # orbital site        1     2     3    4     5     6        n+2    n+3             
    # e- Orbitals:       eBD, eGPD, eEPD, eA1,  eA2,  eA3  ... eAn-1   eAn           
    # 
    ### LEGEND FOR CHAIN STATES u[i]     The first 2^(n+3) + 2^(n+2) elements of u[i] must sum to 1.      eBD = 0 means empty of ligand, eBD = 1 means occupied by oxBD,  eBD = 2 means occupied by reBD
    # bit position: n+3   n+2  n+1    n    n-1   n-2  ...   2      1     na                                    chain states u[i]
    #               eBD, eGPD, eEPD, eA1,  eA2,  eA3  ... eAn-1   eAn    eAt        Decimal Number                u[i] index i      
    #                0     0    0     0     0     0   ...   0      0     na              0                            1                       
    #                0     0    0     0     0     0   ...   0      1     na              1                            2
    #                0     0    0     0     0     0   ...   1      0     na              2                            3
    #                0     0    0     0     0     0   ...   1      1     na              3                            4
    #                .     .    .     .     .     .   ...   .      .      .              .                            .
    #                .     .    .     .     .     .   ...   .      .      .              .                            .
    #                0     1    1     1     1     1   ...   0      1     na          2^(n+2) - 3                  2^(n+2) - 2
    #                0     1    1     1     1     1   ...   1      0     na          2^(n+2) - 2                  2^(n+2) - 1
    #                0     1    1     1     1     1   ...   1      1     na          2^(n+2) - 1                  2^(n+2) + 0
    #                1     0    0     0     0     0   ...   0      0     na          2^(n+2) + 0                  2^(n+2) + 1   
    #                1     0    0     0     0     0   ...   0      1     na          2^(n+2) + 1                  2^(n+2) + 2
    #                1     0    0     0     0     0   ...   1      0     na          2^(n+2) + 2                  2^(n+2) + 3
    #                1     0    0     0     0     0   ...   1      1     na          2^(n+2) + 3                  2^(n+2) + 4
    #                .     .    .     .     .     .   ...   .      .      .              .                            .
    #                .     .    .     .     .     .   ...   .      .      .              .                            .
    #                1     1    1     1     1     1   ...   0      1     na          2^(n+3) - 3                  2^(n+3) - 2   
    #                1     1    1     1     1     1   ...   1      0     na          2^(n+3) - 2                  2^(n+3) - 1
    #                1     1    1     1     1     1   ...   1      1     na          2^(n+3) - 1                  2^(n+3) + 0
    #                2     0    0     0     0     0   ...   0      0     na          2^(n+3) + 0                  2^(n+3) + 1       
    #                2     0    0     0     0     0   ...   0      1     na          2^(n+3) + 1                  2^(n+3) + 2
    #                2     0    0     0     0     0   ...   1      0     na          2^(n+3) + 2                  2^(n+3) + 3
    #                .     .    .     .     .     .   ...   .      .      .              .                            .
    #                .     .    .     .     .     .   ...   .      .      .              .                            .
    #                2     1    1     1     1     1   ...   0      1     na      2^(n+3) + 2^(n+2) - 3         2^(n+3) + 2^(n+2) - 2
    #                2     1    1     1     1     1   ...   1      0     na      2^(n+3) + 2^(n+2) - 2         2^(n+3) + 2^(n+2) - 1
    #                2     1    1     1     1     1   ...   1      1     na      2^(n+3) + 2^(n+2) - 1         2^(n+3) + 2^(n+2) 
    #
    # TOTAL NUMBER OF CHAIN STATES:  2^(n+3) + 2^(n+2)
    #
    #
    # SYSTEM STATE:  u[i]       the first 2^(n+3) + 2^(n+2) elements of u[i] must sum to 1.
    #  extra 4 extra u[i] dimensions to hold liquid concentrations:
    # u[2^(n+3) + 2^(n+2) + 1] is conc of reduced D (reD) in bulk solution    
    # u[2^(n+3) + 2^(n+2) + 2] is conc of oxidized D (oxD) in bulk solution   
    # u[2^(n+3) + 2^(n+2) + 3] is conc of reduced At in bulk solution           
    # u[2^(n+3) + 2^(n+2) + 4] is conc of oxidized At in bulk solution    
    #
    # TOTAL NUMBER OF SYSTEM STATES: 2^(n+3) + 2^(n+2) +4
    #

    n = length(p.V_orbitals) - 4        # n Acceptors:                            A1, A2 ... An
    num_chain_states = 2^(n+3) + 2^(n+2)    # creates an empty, mutable Vector designed to hold tuple items of size two Ints.  Using tuples instead of a Vector of Vectors (Vector{Vector{Int}}) in this specific code provides massive advantages in speed, memory consumption, and code safety.
    num_orbitals = n + 4                # add 4 here due to:     eBD, eGPD, eEPD, A1, A2 ... An, eAt
    num_chain_orbitals = n + 3          # add 3 here due to:     eBD, eGPD, eEPD                             eBD IS ACTUALLY A "TRIT" NOT A BIT BECAUSE --> eBD = 0 means occupied by oxBD,  eBD = 1 means occupied by reBD,   eBD = 2 means empty binding site
    reD_conc_idx  = num_chain_states + 1
    oxD_conc_idx  = num_chain_states + 2
    reAt_conc_idx = num_chain_states + 3
    oxAt_conc_idx = num_chain_states + 4

    # Distances between the closest atoms of the ligand molecules
    d_edges = Matrix{Float64}(zeros(num_chain_orbitals, num_chain_orbitals))
    for i in 1:num_chain_orbitals
        for j in 1:num_chain_orbitals
            d_edges[i,j] = get_closest_distance(p.orbitals_atomic_coords[i], p.orbitals_atomic_coords[j])
        end
    end   


    # Voltage Differences
    V_diff_orbitals = Matrix{Float64}(zeros(num_orbitals, num_orbitals))       # We are simplifying here by assuming the V is the same for eBD and eD 
    for i in 1:num_orbitals       
        for j in 1:num_orbitals
            V_diff_orbitals[i,j] = p.V_orbitals[j] - p.V_orbitals[i] 
        end
    end


    # INITIAL CONDITIONS
    u0 = zeros(num_chain_states + 4)
    # Initial Conditions are at Equilibrium with the binding ligand's liquid concentrations. 
    # [total chains] = [Empty Chain] + [redBD Chain] + [oxBD chain] = [Empty Chain] + [Empty Chain][reD]/K_d_D + [Empty Chain][oxD]/K_d_D , so  [Empty Chain]/[total chain] = 1 / ( 1 + [reD]/K_d_D + [oxD]/K_d_D )  
    fraction_empty = 1 / ( 1 + (p.reD_conc_t0 + p.oxD_conc_t0)/p.K_d_D_bim)                                           # K_d_D is the binding dissociation equilibrium constant for D binding to chain.  For simplicity I'm assuming the the two binding constants (for oxD and reD) and binding kinetic constants are equal
    u0[chain_state_binary_to_idx_rep((2,1,0,zeros(Int,n)...))] = fraction_empty                                   # This is the chain state that has eBD=2 (aka empty binding site) and eGPD=1 and all other bits (eEGP or An) are 0.                          
    # Next, the fraction of bound sites that are reduced is [reBD] / ([oxBD] + [reBD]) = ... 1 / (1 + [oxD]/[reD] * K_d_D/K_d_D)
    fraction_bound_and_reduced = (1 - fraction_empty) / (1 + p.oxD_conc_t0 / p.reD_conc_t0)
    u0[chain_state_binary_to_idx_rep((1,1,0,zeros(Int,n)...))] = fraction_bound_and_reduced                       # This is the chain state with eBD=1 meaning binding occupied with                 # the a << b operation converts a to binary representation then shifts a binary bits to the left (padding its RHS with bit zeros), thus 1<<0 = decimal 1 , 1<<2 = decimal 4
    u0[chain_state_binary_to_idx_rep((0,1,0,zeros(Int,n)...))] = 1 - fraction_empty - fraction_bound_and_reduced  # This is the chain state with BD=1 binding occupied and eBD=0 and eGPD=1 and all else 0.                 # the a << b operation converts a to binary representation then shifts a binary bits to the left (padding its RHS with bit zeros), thus 1<<0 = decimal 1 , 1<<2 = decimal 4
    u0[reD_conc_idx] = p.reD_conc_t0           # total moles of reduced dissolved Donor (bulk concentration)    
    u0[oxD_conc_idx] = p.oxD_conc_t0           # total moles of oxidized dissolved Donor (bulk concentration)    
    u0[reAt_conc_idx] = p.reAt_conc_t0         # total moles of reduced Terminal Acceptor (bulk concentration)    
    u0[oxAt_conc_idx] = p.oxAt_conc_t0         # total moles of oxidized Terminal Acceptor (bulk concentration)   

    #########
    ### Precalculate the indexes of reactants and products for each reaction
    idxs_k_const_rxns_gem       = Tuple{Int, Int, Float64}[]    
    idxs_k_const_excite_eGPD    = Tuple{Int, Int, Float64}[]             # creates an empty, mutable Vector designed to hold tuple items of size two Ints.  Using tuples instead of a Vector of Vectors (Vector{Vector{Int}}) in this specific code provides massive advantages in speed, memory consumption, and code safety.
    idxs_k_const_bind_reD       = Tuple{Int, Int, Float64}[]
    idxs_k_const_bind_oxD       = Tuple{Int, Int, Float64}[]
    idxs_k_const_debind_reD     = Tuple{Int, Int, Float64}[]
    idxs_k_const_debind_oxD     = Tuple{Int, Int, Float64}[]
    idxs_k_const_bimol_At       = Tuple{Int, Int, Float64}[]
    precomputed_interchain_ET   = Tuple{Int, Int, Int, Int, Float64}[]

    ec = -1  # electron's charge
    for rxn_reactant_cs_idx in 1:num_chain_states
        cs = chain_state_idx_to_binary_rep(n,rxn_reactant_cs_idx)
        # eBD = cs[1] ,  eGPD = cs[2] , eEPD = cs[3] , A1 = cs[4] ... An = cs[n+3]

        # Photo-excitation (eGPD -> eEPD)
        if cs[2] == 1 && cs[3] == 0     # eGPD ==1 && eEPD ==0
            rxn_product_cs = collect(cs); rxn_product_cs[2] = 0; rxn_product_cs[3] = 1; rxn_product_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_cs)
            push!(idxs_k_const_excite_eGPD, (rxn_reactant_cs_idx, rxn_product_cs_idx, p.eGPD_cross_section))     # Push a tuple into the vector ,  building a vector containing tuples (each tuple is a pair of integers representing the indexes of)
        end

        # Intra-chain (Geminate) Electron Transfers
        for i in findall(==(1), cs)         # Reactions only exist between an occupied (1) orbital and an unoccupied (0) orbital.
            for j in findall(==(0), cs)     # Reactions only exist between an occupied (1) orbital and an unoccupied (0) orbital.
                rxn_product_cs = collect(cs); rxn_product_cs[i] = 0; rxn_product_cs[j] = 1; rxn_product_cs_idx = chain_state_binary_to_idx_rep(rxn_product_cs)
                if (cs[3] == 1 && cs[2] ==0 && rxn_product_cs[3] == 0 && rxn_product_cs[2] == 1) 
                    push!(idxs_k_const_rxns_gem, (rxn_reactant_cs_idx, rxn_product_cs_idx, p.k_eEPD_eGPD_gem * 1E-12))      # This is a triplet to ground relaxation event, not a Dutton electron transfer event
                else
                    push!(idxs_k_const_rxns_gem, (rxn_reactant_cs_idx, rxn_product_cs_idx, k_dutton_calc(ec * V_diff_orbitals[i,j], d_edges[i,j], p.lambda_dG) * 1E-12))  # DUTTON REACTION COEFFICIENT UNITS: 1/picosecond   The Dutton ruler is written to give units of 1/second, so we multiply by 1E-12 to convert it to 1/picosecond.
                end
            end
        end
        
        # Donor Binding / deBinding reactions between reD or oxD and empty BD sites
        if cs[1] == 2    # the BD binding site is empty (eBD=2)
            rxn_product_cs = collect(cs); rxn_product_cs[1] = 0; rxn_product_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_cs)
            push!(idxs_k_const_bind_oxD, (rxn_reactant_cs_idx, rxn_product_cs_idx, p.k_on_D_bim))
            rxn_product_cs[1] = 1; rxn_product_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_cs)
            push!(idxs_k_const_bind_reD, (rxn_reactant_cs_idx, rxn_product_cs_idx, p.k_on_D_bim))
        end
        if cs[1] == 0 # the BD binding site has oxD in it (eBD=0)
            rxn_product_cs = collect(cs); rxn_product_cs[1] = 2; rxn_product_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_cs)
            push!(idxs_k_const_debind_oxD, (rxn_reactant_cs_idx, rxn_product_cs_idx, p.k_off_D_bim))
        end
        if cs[1] == 1 # the BD binding site has reD in it (eGPD=1)
            rxn_product_cs = collect(cs); rxn_product_cs[1] = 2; rxn_product_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_cs)
            push!(idxs_k_const_debind_reD, (rxn_reactant_cs_idx, rxn_product_cs_idx, p.k_off_D_bim))
        end

        # Bimolecular Electron Transfers to Terminal Acceptor (At)
        if cs[end] == 1 
            rxn_product_cs = collect(cs); rxn_product_cs[end] = 0; rxn_product_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_cs)
            push!(idxs_k_const_bimol_At, (rxn_reactant_cs_idx, rxn_product_cs_idx, k_dutton_calc(ec * V_diff_orbitals[end-1,end], p.d_edge_At, p.lambda_dG) *1E-12))   #UNITS: 1/picosecond   The Dutton ruler is written to give units of 1/second, so we multiply by 1E-12 to convert it to 1/picosecond.
        end

        # The outer loop we're in now is reactant A, so now we need to create a loop to look at each possible reactant B chain.  We only need consider flow of electrons from reactant A to B, since the outer loop will eventually (or already has) choose(n) a chain to test the reverse.
        for rxn_reactant_B_cs_idx in 1:num_chain_states
            for i in findall(==(1), cs)                            # cs is cs_reactant_A.  Reactions only exist between an occupied (1) orbital and an unoccupied (0) orbital.
                for j in findall(==(0), rxn_reactant_B_cs_idx)     # Reactions only exist between an occupied (1) orbital and an unoccupied (0) orbital.
                    cs_reactant_B_cs = chain_state_idx_to_binary_rep(n,rxn_reactant_B_cs_idx)                        
                    rxn_product_A_cs = collect(cs);               rxn_product_A_cs[i] = 0; rxn_product_A_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_A_cs)
                    rxn_product_B_cs = collect(cs_reactant_B_cs); rxn_product_B_cs[j] = 1; rxn_product_B_cs_idx  = chain_state_binary_to_idx_rep(rxn_product_B_cs)
                    push!(cs, (rxn_reactant_cs_idx, rxn_reactant_B_cs_idx, rxn_product_A_cs_idx, rxn_product_B_cs_idx, k_dutton_calc(ec * V_diff_orbitals[i,j], d_edges[i,j], p.lambda_dG) * 1E-12)) # 5. Calculate your rate (K_A * k_rxn) and store the pathway .  Ensure k_obs has units of M^-1 s^-1 and is scaled to picoseconds if your tspan requires it
                end
            end
        end

    end

    # Parameters for ODE Solver.  Note: the Dutton ruler is written to give units of 1/second, so we multiply by 1E-12 to convert it to 1/picosecond.
    ode_p = (
        p_in                        = p,
        num_orbitals                = num_orbitals,
        num_chain_state_bits        = num_chain_orbitals,
        num_chain_states            = num_chain_states,
        reD_conc_idx                = reD_conc_idx,
        oxD_conc_idx                = oxD_conc_idx,
        reAt_conc_idx               = reAt_conc_idx,
        oxAt_conc_idx               = oxAt_conc_idx,
        idxs_k_const_rxns_gem       = idxs_k_const_rxns_gem,     
        idxs_k_const_excite_eGPD    = idxs_k_const_excite_eGPD, 
        idxs_k_const_bind_reD       = idxs_k_const_bind_reD,    
        idxs_k_const_bind_oxD       = idxs_k_const_bind_oxD,    
        idxs_k_const_debind_reD     = idxs_k_const_debind_reD,  
        idxs_k_const_debind_oxD     = idxs_k_const_debind_oxD,  
        idxs_k_const_bimol_At       = idxs_k_const_bimol_At,    
        precomputed_interchain_ET   = precomputed_interchain_ET )


    ### Utilize an ODE Solver
    # u is a vector with elements representing ensemble-averagd occupancy of each protein chain state (and in the case of u[49] or u[51] a concentration of solutes in the bulk solution)
    # dudt is a vector with elements representing the change in u that

    tspan = (0.0, p.tspan)       # time span in picoseconds
    problem = ODEProblem(dudt!, u0, tspan, ode_p)
    function check_neg(u, ode_p, t); return any(x -> x < 0.0, u); end     # Numerical Stability Enhancer: Reject Negative Steps. This tells the solver: "If any value in u goes negative, reject the step and try smaller dt"
    println("Solving with Rodas5P (Stiff+Robust)...")
    solution = solve(problem, Rodas5P(), reltol=1e-8, abstol=1e-8, isoutofdomain=check_neg, tstops=[1000])     # Rodas5P is more robust than RadauIIA5 for these specific biological/chemical systems.       tstops=[1.0] ensures pulse center is hit      # isoutofdomain=check_neg prevents negative pops  
    println("Done.")

    return solution, ode_p
end













using TOML

function build_input_parameters(config_path::String)

    # 1. Parse the TOML file into a Julia Dictionary
    cfg = TOML.parsefile(config_path)

    # 2. Handle Coordinates Extraction
    if cfg["ligands"]["cif_pdb_filename_or_manual"] == "manual"
        # Convert parsed TOML arrays into Julia strongly-typed Vectors
        raw_coords = cfg["ligands"]["cif_pdb_identifiers_or_manual_data"]
        atomic_coords = [[Float64.(atom) for atom in molecule] for molecule in raw_coords]
    else
        atomic_coords = extract_ligands_atomic_coords(cfg["ligands"]["cif_pdb_filename_or_manual"],cfg["ligands"]["cif_pdb_identifiers_or_manual_data"])
    end

    orbitals_atomic_coords = Vector{Vector{Float64}}()
    push!(orbitals_atomic_coords, atomic_coords[1])     # this is the eBD  orbital
    push!(orbitals_atomic_coords, atomic_coords[2])     # this is the eGPD orbital
    push!(orbitals_atomic_coords, atomic_coords[2])     # this is the eEPD orbital
    for i in 3:length(atomic_coords)                      # these are the eA1 ... eAn orbitals
        push!(orbitals_atomic_coords, atomic_coords[i])   # these are the eA1 ... eAn orbitals
    end

    V_orbitals = Float64.(cfg["thermodynamics"]["V_orbitals"])
    N_avagadro = 6.022e23

    # 3. Calculate Diffusion & Binding Kinetics
    D_chain          = Float64(cfg["diffusion"]["D_chain"])
    D_D              = Float64(cfg["diffusion"]["D_D"])
    D_At             = Float64(cfg["diffusion"]["D_At"])
    radius_gyr_chain = Float64(cfg["geometry"]["radius_gyr_chain"])
    radius_gyr_D     = Float64(cfg["geometry"]["radius_gyr_D"])
    radius_gyr_At    = Float64(cfg["geometry"]["radius_gyr_At"])

    # Donor Binding parameters
    K_d_D_bim   = Float64(cfg["thermodynamics"]["K_d_D_bim"])
    k_on_D_bim  = 4 * π * N_avagadro * (D_D + D_chain) * (radius_gyr_chain + radius_gyr_D)/2 * 1e-12 * 1e-10 * 1000
    k_off_D_bim = K_d_D_bim * k_on_D_bim

    # Bimolecular chain to At Binding parameters
    K_a_At_bim   = 4/3 * π * N_avagadro * ((radius_gyr_chain + radius_gyr_At)/2)^3 * 1E-30 * 1000 
    K_d_At_bim   = 1/K_a_At_bim
    k_on_At_bim  = 4 * π * N_avagadro * (D_At + D_chain) * (radius_gyr_chain + radius_gyr_At)/2 * 1e-12 * 1e-10 * 1000
    k_off_At_bim = K_d_At_bim * k_on_At_bim

    # Bimolecular chain to chain Binding parameters
    K_a_chains_bim   = 4/3 * π * N_avagadro * ((radius_gyr_chain + radius_gyr_chain)/2)^3 * 1E-30 * 1000 
    K_d_chains_bim   = 1/K_a_chains_bim
    k_on_chains_bim  = 4 * π * N_avagadro * (D_At + D_chain) * (radius_gyr_chain + radius_gyr_chain)/2 * 1e-12 * 1e-10 * 1000
    k_off_chains_bim = K_d_chains_bim * k_on_chains_bim

    # 4. Construct the NamedTuple `p`
    p = (
        N_avagadro                = N_avagadro,                                  #[cite: 1]
        conc_chains               = Float64(cfg["environment"]["conc_chains"]),  # concentration of chains in molesns, UNITS: moles/L[cite: 1]
        mixture_volume            = Float64(cfg["environment"]["mixture_volume"]), # volume of the mixture, UNITS: Liters     I assume the volume of the mixture is exactly the liquid volume that is illuminated by light.[cite: 1]
        illuminated_cross_section = Float64(cfg["environment"]["illuminated_cross_section"]), # UNITS: m^2    this is 3.3 cm x 3.3 cm[cite: 1]
        orbitals_atomic_coords    = orbitals_atomic_coords,                      # coordinates of atoms in this electron acceptor molecule, UNITS: Angstroms[cite: 1]
        V_orbitals                = V_orbitals,                                  # midpoint Voltages[cite: 1]
        d_bim                     = Float64(cfg["geometry"]["d_bim"]),           # distance between molecules during bimolecular electron transfer, UNITS: Angstroms[cite: 1]
        reD_conc_t0               = Float64(cfg["environment"]["reD_conc_t0"]),  # total concentration of reduced dissolved donor molecules, UNITS: moles/L[cite: 1]
        oxD_conc_t0               = Float64(cfg["environment"]["oxD_conc_t0"]),  # total concentration of oxidized dissolved donor molecules, UNITS: moles/L[cite: 1]
        K_d_D_bim                 = K_d_D_bim,                                   # donor binding dissociation equilibrium constant, k_d = [liq D] [liq Protein w empty site] / [bound BD]        for simplicity I'm assuming the the two binding constants  (for oxD and reD) and binding kinetic constants are equal[cite: 1]
        k_on_D_bim                = k_on_D_bim,                                  # typical diffusion limited binding rate, UNITS: 1/M /s          for simplicity I'm assuming the the two binding constants  (for oxD and reD) and binding kinetic constants are equal[cite: 1]
        k_off_D_bim               = k_off_D_bim,                                 # donor debinding rate, UNITS: 1/s                              for simplicity I'm assuming the the two binding constants and binding kinetic constants are equal[cite: 1]
        K_d_At_bim                = K_d_At_bim,                                  # At binding dissociation equilibrium constant, k_d = [liq D] [liq Protein w empty site] / [bound BD][cite: 1]
        k_on_At_bim               = k_on_At_bim,                                 # typical diffusion limited binding rate, UNITS: 1/M /s[cite: 1]
        k_off_At_bim              = k_off_At_bim,                                # At debinding rate, UNITS: 1/s[cite: 1]
        K_d_chains_bim            = K_d_chains_bim,                              # chain to chain binding dissociation equilibrium constant, k_d = [liq chain] [liq chain w empty site] / [bound chain][cite: 1]
        k_on_chains_bim           = k_on_chains_bim,                             # typical diffusion limited binding rate, UNITS: 1/M /s[cite: 1]
        k_off_chains_bim          = k_off_chains_bim,                            # chain to chain debinding rate, UNITS: 1/s[cite: 1]
        radius_chain              = Float64(cfg["geometry"]["radius_chain"]),    # radius of the chain, UNITS: Angstroms   From this radius we calculate the reaction kinetic Smoluchowski k_on and Eigen-Fuoss reaction association constant K_A[cite: 1]
        radius_D                  = Float64(cfg["geometry"]["radius_D"]),        # radius of the donor molecule, UNITS: Angstroms[cite: 1]
        radius_At                 = Float64(cfg["geometry"]["radius_At"]),       # radius of the terminal acceptor molecule, UNITS: Angstroms[cite: 1]
        d_edge_At                 = Float64(cfg["geometry"]["d_edge_At"]),       # distance between the terminal acceptor and A2 on the chain, UNITS: Angstroms[cite: 1]
        reAt_conc_t0              = Float64(cfg["environment"]["reAt_conc_t0"]), # concentration of reduced dissolved terminal acceptor molecules, UNITS: moles/L[cite: 1]
        oxAt_conc_t0              = Float64(cfg["environment"]["oxAt_conc_t0"]), # total concentration of oxidized dissolved terminal acceptor molecules, UNITS: moles/L[cite: 1]
        lambda_dG                 = Float64(cfg["thermodynamics"]["lambda_dG"]), # Photo-excitation reorganization energy (eV)[cite: 1]
        k_eEPD_eGPD_gem           = Float64(cfg["thermodynamics"]["k_eEPD_eGPD_gem"]), # triplet state relaxation rate of excited photo-absorber orbital to ground state orbital on photo-absorber,  UNITS: 1/seconds[cite: 1]
        pulse_type                = Int(cfg["pulse"]["pulse_type"]),             # select gaussian or square or step pulse, or no pulse     #Pulse Type    0 = no pulse,          1 = step pulse,             2=square,                   3=gaussian[cite: 1]
        pulse_param1              = Float64(cfg["pulse"]["pulse_param1"]),       # amplitude photon flux @ absorb wavelength        UNITS: #/m^2/s.        #Pulse parameter 1       isn't used,            Amplitude (k_max),          Amplitude (k_max),          Amplitude (k_max) - Very high rate for short time[cite: 1]
        pulse_param2              = Float64(cfg["pulse"]["pulse_param2"]),       # onset time (ps)                                                         #Pulse parameter 2       isn't used,            onset time (ps),            onset time (ps),            Center (t = 1.0 ps)[cite: 1]
        pulse_param3              = Float64(cfg["pulse"]["pulse_param3"]),       # stop time or width of the pulse (ps)                                    #Pulse parameter 3       isn't used,            isn't used,                 stop time (ps),             Width (sigma = 0.1 ps)[cite: 1]
        eGPD_cross_section        = Float64(cfg["pulse"]["eGPD_cross_section"]), # cross-section of a single eGPD orbital. UNITS: m^2    We assume no molecules eclipse other molecules in the beam path of light.      Cross section is converted to absorption coefficient (units 1/M/cm) via:  Abs Coeff = Cross Sec x N_Avagadro / ln(10) / 1000  (where 1000 converts L to cm^3)[cite: 1]
        tspan                     = Float64(cfg["pulse"]["tspan"]),              # total simulation time (ps)[cite: 1]
        verbose                   = Bool(cfg["output"]["verbose"])               # whether to print output to the console or keep quiet.  A TOML output file is ALWAYS created either way.
        )

    return p
end


























function plot_and_data_dump(solution, ode_p, p)

    ### Plot it all up 
    mkpath("output_data_n_figs") #this command automatically checks if the folder already exists

    #     # Distances from center of ligands to centers of ligands
    # d_centers = Matrix{Float64}(zeros(num_chain_orbitals, num_chain_orbitals))
    # for i in 1:num_chain_orbitals
    #     for j in 1:num_chain_orbitals
    #         d_centers[i,j] = norm(loc_centers[i] - loc_centers[j])
    #     end
    # end

    # Recalculate Geometry Centers for plotting
    loc_centers = [get_geometric_center(coords) for coords in p.orbitals_atomic_coords]
    d_centers_BD_PD = norm(loc_centers[1] - loc_centers[2])
    d_centers_PD_eA1 = norm(loc_centers[2] - loc_centers[4])
    d_centers_eA1_eA2 = norm(loc_centers[4] - loc_centers[5])

    # Voltage Differences
    V_diff_orbitals = Matrix{Float64}(zeros(ode_p.num_orbitals, ode_p.num_orbitals))       # We are simplifying here by assuming the V is the same for eBD and eD 
    for i in 1:ode_p.num_orbitals       
        for j in 1:ode_p.num_orbitals
            V_diff_orbitals[i,j] = p.V_orbitals[j] - p.V_orbitals[i] 
        end
    end

    # Distances between the closest atoms of the ligand molecules
    d_edges = Matrix{Float64}(zeros(ode_p.num_orbitals - 1, ode_p.num_orbitals - 1))
    for i in 1:ode_p.num_orbitals - 1
        for j in 1:ode_p.num_orbitals - 1
            d_edges[i,j] = get_closest_distance(p.orbitals_atomic_coords[i], p.orbitals_atomic_coords[j])
        end
    end   


    # Electron Transfer Kinetics Intra-Chain (aka Geminate). 1st Order Kinetics
    ec = -1  # electron's charge
    dutton_e_transfer_k_matrix_gem = Matrix{Float64}(zeros(ode_p.num_orbitals - 1, ode_p.num_orbitals - 1))  # -1 because we don't want to include the eAt orbital in the matrix of geminate k_dutton coefficients
    for i in 1:ode_p.num_orbitals - 1
        for j in 1:ode_p.num_orbitals - 1
            dutton_e_transfer_k_matrix_gem[i,j] = k_dutton_calc(ec * V_diff_orbitals[i,j], d_edges[i,j], p.lambda_dG) *1E-12 # UNITS: 1/picosecond   The Dutton ruler is written to give units of 1/second, so we multiply by 1E-12 to convert it to 1/picosecond.
            if i==j; dutton_e_transfer_k_matrix_gem[i,j] = 0; end
        end
    end


    time = solution.t
    n = ode_p.num_orbitals - 4

    # Tracking arrays setup
    flow_vs_t_10xx_exc        = zeros(length(time))   
    flow_vs_t_01xx_relax      = zeros(length(time))   
    flow_vs_t_eEPD_eA1        = zeros(length(time))   
    flow_vs_t_eA1_eEPD        = zeros(length(time))   
    flow_vs_t_eA1_eA2         = zeros(length(time))   
    flow_vs_t_eA2_eA1         = zeros(length(time))   
    flow_vs_t_eA1_eGPD        = zeros(length(time))   
    flow_vs_t_eA2_eGPD        = zeros(length(time))   
    flow_vs_t_eA2_At_pCh      = zeros(length(time))   
    flow_vs_t_eA2_At_p_L      = zeros(length(time))   

    # Note: Bimol flows for D->eGPD etc., are omitted here to save space as their tracking is highly dependent on how K_d translates to your desired visual. Added zeros arrays below to prevent plotting errors.
    flow_vs_t_D_eGPD_pCh = zeros(length(time)); flow_vs_t_eEPD_hD_pCh = zeros(length(time)); flow_vs_t_eA1_hD_pCh = zeros(length(time)); flow_vs_t_eA2_hD_pCh = zeros(length(time))
    flow_vs_t_D_eGPD_p_L = zeros(length(time)); flow_vs_t_eEPD_hD_p_L = zeros(length(time)); flow_vs_t_eA1_hD_p_L = zeros(length(time)); flow_vs_t_eA2_hD_p_L = zeros(length(time))
    flow_vs_t_eEPD_eGPD_bim1 = zeros(length(time)); flow_vs_t_eEPD_eGPD_bim2 = zeros(length(time))
    flow_vs_t_eA1_eGPD_bim1 = zeros(length(time)); flow_vs_t_eA1_eGPD_bim2 = zeros(length(time))
    flow_vs_t_eA2_eGPD_bim1 = zeros(length(time)); flow_vs_t_eA2_eGPD_bim2 = zeros(length(time))

    for (idx_t, t) in enumerate(time)
        u = solution.u[idx_t]
        
        # Calculate pulse intensity
        k_exc_vs_t = 0.0
        if ode_p.p_in.pulse_type == 1; if t >= ode_p.p_in.pulse_param2; k_exc_vs_t = ode_p.p_in.pulse_param1; end
        elseif ode_p.p_in.pulse_type == 2; if t >= ode_p.p_in.pulse_param2 && t <= ode_p.p_in.pulse_param3; k_exc_vs_t = ode_p.p_in.pulse_param1; end
        elseif ode_p.p_in.pulse_type == 3; k_exc_vs_t = gaussian_pulse(t, ode_p.p_in.pulse_param1, ode_p.p_in.pulse_param2, ode_p.p_in.pulse_param3)
        end
        
        # Track Excitation
        for (u_idxs_reactant, u_idxs_rxn_product, cross_section) in ode_p.idxs_k_const_excite_eGPD
            flow_vs_t_10xx_exc[idx_t] += k_exc_vs_t * cross_section * 1E-12 * u[u_idxs_reactant]
        end    

        # Track Geminate Transfers
        for (u_idxs_reactant, u_idxs_rxn_product, k_const) in ode_p.idxs_k_const_rxns_gem
            BD, eBD, eGPD, eEPD, eAn = chain_state_idx_to_binary_rep(u_idxs_reactant, n)
            BD_d, eBD_d, eGPD_d, eEPD_d, eAn_d = chain_state_idx_to_binary_rep(u_idxs_rxn_product, n)
            
            if eEPD == 1 && eEPD_d == 0 && eAn[1] == 0 && eAn_d[1] == 1
                flow_vs_t_eEPD_eA1[idx_t] += k_const * u[u_idxs_reactant]
            elseif eAn[1] == 1 && eAn_d[1] == 0 && eEPD == 0 && eEPD_d == 1
                flow_vs_t_eA1_eEPD[idx_t] += k_const * u[u_idxs_reactant]
            elseif length(eAn) >= 2 && eAn[1] == 1 && eAn_d[1] == 0 && eAn[2] == 0 && eAn_d[2] == 1
                flow_vs_t_eA1_eA2[idx_t] += k_const * u[u_idxs_reactant]
            elseif length(eAn) >= 2 && eAn[2] == 1 && eAn_d[2] == 0 && eAn[1] == 0 && eAn_d[1] == 1
                flow_vs_t_eA2_eA1[idx_t] += k_const * u[u_idxs_reactant]
            elseif eAn[1] == 1 && eAn_d[1] == 0 && eGPD == 0 && eGPD_d == 1
                flow_vs_t_eA1_eGPD[idx_t] += k_const * u[u_idxs_reactant]
            elseif length(eAn) >= 2 && eAn[2] == 1 && eAn_d[2] == 0 && eGPD == 0 && eGPD_d == 1
                flow_vs_t_eA2_eGPD[idx_t] += k_const * u[u_idxs_reactant]
            elseif eEPD == 1 && eEPD_d == 0 && eGPD == 0 && eGPD_d == 1
                flow_vs_t_01xx_relax[idx_t] += k_const * u[u_idxs_reactant]
            end
        end
        
        # Track Bimol Outflow to Terminal Acceptor
        oxAt_conc = u[ode_p.oxAt_conc_idx]
        for (u_idxs_reactant, u_idxs_rxn_product, k_const) in ode_p.idxs_k_const_bimol_At
            flow_pCh = k_const * oxAt_conc * u[u_idxs_reactant]
            flow_vs_t_eA2_At_pCh[idx_t] += flow_pCh
            flow_vs_t_eA2_At_p_L[idx_t] += flow_pCh * p.conc_chains
        end
    end



    # # Electron Transfer Kinetics Inter-Chain (aka Bimolecular). 2nd Order Kinetics
    # dutton_e_transfer_k_matrix_bim = Matrix{Float64}(zeros(ode_p.num_orbitals + 1, ode_p.num_orbitals + 1))   # We are simplifying here by assuming the k_dutton_bim is the same for eBD and eD for bimolecular reactions
    # for i in 1:ode_p.num_orbitals + 1
    #     for j in 1:ode_p.num_orbitals + 1
    #         dutton_e_transfer_k_matrix_bim[i,j] = k_dutton_calc(ec * V_diff_bim[i,j], ode_p.d_bim, p.lambda_dG) *1E-12 # UNITS: 1/picosecond   The Dutton ruler is written to give units of 1/second, so we multiply by 1E-12 to convert it to 1/picosecond.
    #         if i==j; dutton_e_transfer_k_matrix_bim[i,j] = 0; end
    #     end
    # end



    # Fig 0: plot the voltage and location of each active molecule
    fig0_han, ax0_han = plt.subplots(figsize=(10, 6))
    x_coords = [-d_centers_BD_PD, 0.0, 0.0, d_centers_PD_eA1, d_centers_PD_eA1 + d_centers_eA1_eA2]
    y_coords = [p.V_orbitals[1], p.V_orbitals[2], p.V_orbitals[3], p.V_orbitals[4], p.V_orbitals[5]]
    state_labels = ["eBD", "eGPD", "eEPD", "A1", "A2"]
    ax0_han.scatter(x_coords, y_coords, marker="_", s=2500, lw=4, color="black", zorder=3)
    ax0_han.invert_yaxis()

    # Annotate each horizontal line with its corresponding molecule/state name
    for i in 1:length(x_coords)
        ax0_han.annotate(state_labels[i], (x_coords[i], y_coords[i]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
    end

    ax0_han.xaxis.set_label_text("Location / Å")
    ax0_han.yaxis.set_label_text("Voltage (NHE) / V")
    ax0_han.set_title("Midpoint Voltage vs. Distance")
    ax0_han.grid(true, alpha=0.3, zorder=0)
    fig0_han.savefig("output_data_n_figs/fig0.png", bbox_inches="tight")



    # Proper Vector Initialization for Ensembles
    ens_ave_oxBD  = zeros(length(time))
    ens_ave_reBD = zeros(length(time))
    ens_ave_eGPD = zeros(length(time))
    ens_ave_eEPD = zeros(length(time))
    ens_ave_eA1  = zeros(length(time))
    ens_ave_eA2  = zeros(length(time))

    # Calculate the ensemble averaged occupancies
    for i in 1:ode_p.num_chain_states
        cs_binary = chain_state_idx_to_binary_rep(ode_p.num_orbitals - 4, i)
        if cs_binary[1] == 0; ens_ave_oxBD .+= solution[i, :]; end
        if cs_binary[1] == 1; ens_ave_reBD .+= solution[i, :]; end
        if cs_binary[2] == 1; ens_ave_eGPD .+= solution[i, :]; end
        if cs_binary[3] == 1; ens_ave_eEPD .+= solution[i, :]; end
        if cs_binary[4] == 1; ens_ave_eA1  .+= solution[i, :]; end
        if cs_binary[5] == 1; ens_ave_eA2  .+= solution[i, :]; end
    end


    photon_flux_time_series = zeros(length(time))
    for (i, t) in enumerate(time)
        if p.pulse_type == 1 
            if t >= p.pulse_param2; photon_flux_time_series[i] = p.pulse_param1; end
        elseif p.pulse_type == 2 
            if t >= p.pulse_param2 && t <= p.pulse_param3; photon_flux_time_series[i] = p.pulse_param1; end
        elseif p.pulse_type == 3 
            photon_flux_time_series[i] = gaussian_pulse(t, p.pulse_param1, p.pulse_param2, p.pulse_param3)
        end
    end
    total_photons = 0.0
    Num_Excitations = 0.0
    for i in 2:length(time)
        dt_ps = time[i] - time[i-1]
        dt_s = dt_ps * 1e-12  
        avg_flux = (photon_flux_time_series[i] + photon_flux_time_series[i-1]) / 2.0   # Integrate Fluence (photons / m^2)
        total_photons += avg_flux * dt_s
        avg_exc = (flow_vs_t_10xx_exc[i] + flow_vs_t_10xx_exc[i-1]) / 2.0    # Integrate total fractional excitations (ps^-1 * ps = dimensionless fraction)
        Num_Excitations += avg_exc * dt_ps
    end

        
    pulse_profile = [gaussian_pulse(t, p.pulse_param1, p.pulse_param2, p.pulse_param3) for t in time]

    # Fig 1: plot the occupancy of each state over time
    fig1_han, ax11_han = plt.subplots(figsize=(22, 6))
    ax12_han = ax11_han.twinx()
    ax11_han.plot(time, ens_ave_oxBD, linestyle="dashed", label="oxBD", lw=2)
    ax11_han.plot(time, ens_ave_reBD, linestyle="dashed", label="reBD", lw=2)
    ax11_han.plot(time, ens_ave_eGPD, linestyle="dashed", label="eGPD", lw=2)
    ax11_han.plot(time, ens_ave_eEPD, linestyle="dashed", label="eEPD", lw=2)
    ax11_han.plot(time, ens_ave_eA1, linestyle="dotted", label="A1", lw=2)
    ax11_han.plot(time, ens_ave_eA2, linestyle="dashdot", label="A2", lw=2)
    ax12_han.fill_between(time, pulse_profile, color="gray", alpha=0.18, label="Photon Pulse")
    lines11, labels11 = ax11_han.get_legend_handles_labels()
    lines12, labels12 = ax12_han.get_legend_handles_labels()
    ax11_han.xaxis.set_label_text("time / ps")
    ax12_han.yaxis.set_label_text("photon flux / photons/m^2")
    ax11_han.yaxis.set_label_text("occupancy")
    ax11_han.legend(vcat(lines11,lines12), vcat(labels11,labels12), loc="best")
    ax11_han.grid(true, alpha=0.3)
    fig1_han.savefig("output_data_n_figs/fig1.png", bbox_inches="tight")


    # Fig 2: plot the occupancy of each state over time
    fig2_han, ax21_han = plt.subplots(figsize=(22, 6))
    ax22_han = ax21_han.twinx()
    ax21_han.plot(time, ens_ave_oxBD, linestyle="dashed", label="oxBD", lw=2)
    ax21_han.plot(time, ens_ave_reBD, linestyle="dashed", label="reBD", lw=2)
    ax21_han.plot(time, ens_ave_eGPD, linestyle="dashed", label="eGPD", lw=2)
    ax21_han.plot(time, ens_ave_eEPD, linestyle="dashed", label="eEPD", lw=2)
    ax21_han.plot(time, ens_ave_eA1, linestyle="dotted", label="A1", lw=2)
    ax21_han.plot(time, ens_ave_eA2, linestyle="dashdot", label="A2", lw=2)
    ax22_han.fill_between(time, pulse_profile, color="gray", alpha=0.18, label="Photon Pulse")
    ax21_han.set_xlim([0, time[end]/10])
    ax22_han.set_xlim([0, time[end]/10])
    lines21, labels21 = ax21_han.get_legend_handles_labels()
    lines22, labels22 = ax22_han.get_legend_handles_labels()
    ax21_han.xaxis.set_label_text("time / ps")
    ax22_han.yaxis.set_label_text("photon flux / photons/m^2")
    ax21_han.yaxis.set_label_text("occupancy")
    ax21_han.legend(vcat(lines21,lines22), vcat(labels21,labels22), loc="best")
    ax21_han.grid(true, alpha=0.3)
    fig2_han.savefig("output_data_n_figs/fig2.png", bbox_inches="tight")


    # Fig 3: plot the concentration of each bulk solute molecule over time
    fig3_han, ax31_han = plt.subplots(figsize=(18, 6))
    ax32_han = ax31_han.twinx()
    ax31_han.plot(time, solution[ode_p.reD_conc_idx,:], linestyle="dashed", label="D", lw=2)
    ax32_han.plot(time, solution[ode_p.reAt_conc_idx,:], linestyle="dotted", label="At", lw=2, color="r")
    ax31_han.set_ylim([0, p.oxD_conc_t0 + p.reD_conc_t0] )
    ax32_han.set_ylim([0, p.oxAt_conc_t0 + p.reAt_conc_t0])
    lines21, labels21 = ax31_han.get_legend_handles_labels()
    lines22, labels22 = ax32_han.get_legend_handles_labels()
    ax31_han.xaxis.set_label_text("time / ps")
    ax32_han.yaxis.set_label_text("concentration / M")
    ax31_han.yaxis.set_label_text("concentration / M")
    ax32_han.legend(vcat(lines21,lines22), vcat(labels21,labels22), loc="best")
    ax31_han.grid(true, alpha=0.3)
    fig3_han.savefig("output_data_n_figs/fig3.png", bbox_inches="tight")

    fig4_han, ax4_han = plt.subplots(figsize=(18, 6))
    ax4_han.plot(time, flow_vs_t_10xx_exc,     linestyle="dashed", label="flow_10xx_exc", lw=2)
    ax4_han.plot(time, flow_vs_t_01xx_relax,   linestyle="dashed", label="flow_01xx_relax", lw=2)
    ax4_han.plot(time, flow_vs_t_eEPD_eA1,     linestyle="dashed", label="flow_eEPD_eA1", lw=2)
    ax4_han.plot(time, flow_vs_t_eA1_eEPD,     linestyle="dashed", label="flow_eA1_eEPD", lw=2)
    ax4_han.plot(time, flow_vs_t_eA1_eA2,      linestyle="dashed", label="flow_eA1_eA2", lw=2)
    ax4_han.plot(time, flow_vs_t_eA2_eA1,      linestyle="dashed", label="flow_eA2_eA1", lw=2)
    ax4_han.plot(time, flow_vs_t_eA1_eGPD,     linestyle="dashed", label="flow_eA1_eGPD", lw=2)
    ax4_han.plot(time, flow_vs_t_eA2_eGPD,     linestyle="dashed", label="flow_eA2_eGPD", lw=2)
    ax4_han.xaxis.set_label_text("time / ps")
    ax4_han.yaxis.set_label_text("flow of occupancy / (1/ps)")
    ax4_han.legend()
    fig4_han.savefig("output_data_n_figs/fig4.png", bbox_inches="tight")

    fig5_han, ax5_han = plt.subplots(figsize=(18, 6))
    ax5_han.plot(time, flow_vs_t_eA2_At_pCh,  linestyle="dashed", label="flow_eA2_At / Chain", lw=2)
    ax5_han.plot(time, flow_vs_t_D_eGPD_pCh,  linestyle="dashed", label="flow_D_eGPD / Chain", lw=2)
    ax5_han.plot(time, flow_vs_t_eEPD_hD_pCh, linestyle="dashed", label="flow_eEPD_hD / Chain", lw=2)
    ax5_han.plot(time, flow_vs_t_eA1_hD_pCh,  linestyle="dashed", label="flow_eA1_hD / Chain", lw=2)
    ax5_han.plot(time, flow_vs_t_eA2_hD_pCh,  linestyle="dashed", label="flow_eA2_hD / Chain", lw=2)
    ax5_han.xaxis.set_label_text("time / ps")
    ax5_han.yaxis.set_label_text("flow of occupancy / (1/ps)")
    ax5_han.legend()
    fig5_han.savefig("output_data_n_figs/fig5.png", bbox_inches="tight")

    fig6_han, ax6_han = plt.subplots(figsize=(18, 6))
    ax6_han.plot(time, flow_vs_t_eA2_At_p_L,  linestyle="dashed", label="flow_eA2_At / L", lw=2)
    ax6_han.plot(time, flow_vs_t_D_eGPD_p_L,  linestyle="dashed", label="flow_D_eGPD / L", lw=2)
    ax6_han.plot(time, flow_vs_t_eEPD_hD_p_L, linestyle="dashed", label="flow_eEPD_hD / L", lw=2)
    ax6_han.plot(time, flow_vs_t_eA1_hD_p_L,  linestyle="dashed", label="flow_eA1_hD / L", lw=2)
    ax6_han.plot(time, flow_vs_t_eA2_hD_p_L,  linestyle="dashed", label="flow_eA2_hD / L", lw=2)
    ax6_han.xaxis.set_label_text("time / ps")
    ax6_han.yaxis.set_label_text("flow of concentration / (M/ps)")
    ax6_han.legend()
    fig6_han.savefig("output_data_n_figs/fig6.png", bbox_inches="tight")

    fig7_han, ax7_han = plt.subplots(figsize=(18, 6))
    ax7_han.plot(time, flow_vs_t_eEPD_eGPD_bim1, linestyle="dashed", label="flow_eEPD_eGPD_bim1", lw=2)
    ax7_han.plot(time, flow_vs_t_eEPD_eGPD_bim2, linestyle="dashed", label="flow_eEPD_eGPD_bim2", lw=2)
    ax7_han.plot(time, flow_vs_t_eA1_eGPD_bim1,  linestyle="dashed", label="flow_eA1_eGPD_bim1", lw=2)
    ax7_han.plot(time, flow_vs_t_eA1_eGPD_bim2,  linestyle="dashed", label="flow_eA1_eGPD_bim2", lw=2)
    ax7_han.plot(time, flow_vs_t_eA2_eGPD_bim1,  linestyle="dashed", label="flow_eA2_eGPD_bim1", lw=2)
    ax7_han.plot(time, flow_vs_t_eA2_eGPD_bim2,  linestyle="dashed", label="flow_eA2_eGPD_bim2", lw=2)
    ax7_han.xaxis.set_label_text("time / ps")
    ax7_han.yaxis.set_label_text("flow of occupancy / (1/ps)")
    ax7_han.legend()
    fig7_han.savefig("output_data_n_figs/fig7.png", bbox_inches="tight")

    figDk_han, axDk_han = plt.subplots(figsize=(18, 6))
    dV = range(0, stop=2, length=100)
    distance = range(5, stop=20, length=100)
    k = k_dutton_downhill.(-dV', distance, p.lambda_dG)
    mpl_colors = PyCall.pyimport("matplotlib.colors")
    pcm = axDk_han.pcolormesh(dV, distance, k, norm=mpl_colors.LogNorm(vmin=minimum(k), vmax=maximum(k)), shading="auto", cmap="viridis")
    figDk_han.colorbar(pcm, ax=axDk_han, label="k dutton kinetics / (1/s)")
    axDk_han.set_title("Dutton kinetics downhill")
    axDk_han.xaxis.set_label_text("ΔV / eV")
    axDk_han.yaxis.set_label_text("Distance / Å")
    figDk_han.savefig("output_data_n_figs/figDk.png", bbox_inches="tight")

    figDkU_han, axDkU_han = plt.subplots(figsize=(18, 6))
    dV = range(0, stop=1, length=100)
    distance = range(5, stop=20, length=100)
    k = k_dutton_uphill.(dV', distance, p.lambda_dG)
    pcm = axDkU_han.pcolormesh(dV, distance, k, norm=mpl_colors.LogNorm(vmin=minimum(k), vmax=maximum(k)), shading="auto", cmap="viridis")
    figDkU_han.colorbar(pcm, ax=axDkU_han, label="k dutton kinetics uphill / (1/s)")
    axDkU_han.set_title("Dutton kinetics uphill")
    axDkU_han.xaxis.set_label_text("ΔV / eV")
    axDkU_han.yaxis.set_label_text("Distance / Å") 
    figDkU_han.savefig("output_data_n_figs/figDkU.png", bbox_inches="tight")






    moles_incident_photons = (total_photons * p.illuminated_cross_section) / p.N_avagadro
    moles_A2_produced = ens_ave_eA2[end] * p.conc_chains * p.mixture_volume
    total_efficiency = moles_A2_produced / moles_incident_photons
    quantum_efficiency = ens_ave_eA2[end] / Num_Excitations   # Calculate Quantum Efficiency (Fraction of chains producing A2 / Fraction of chains excited) Because both are measured per chain, volume and concentration perfectly cancel out.
    if p.verbose
        println(@sprintf("Total Incident Fluence: %1.3e photons/m^2", total_photons))
        println(@sprintf("Number of Excitations: %1.3e", Num_Excitations))
        println(@sprintf("Number of Relaxations: %1.3e", Num_Excitations))
        println(@sprintf("Total Efficiency: %1.3e", total_efficiency))
        println(@sprintf("Quantum Efficiency: %1.3e", quantum_efficiency))
    end
    data_dump = Dict(
        "moles_incident_photons" => moles_incident_photons,
        "moles_A2_produced" => moles_A2_produced,
        "total_efficiency" => total_efficiency,
        "quantum_efficiency" => quantum_efficiency
    )
    open("output_data_n_figs/results.toml", "w") do io
        TOML.print(io, data_dump)
    end


    # write(f, "Fluence,Excitations,Relaxations,Total_Efficiency,Quantum_Efficiency\n"); end                           # If it's a brand new file, write the header row first 
    # write(f, "$(total_photons),$(Num_Excitations),$(Num_Excitations),$(total_efficiency),$(quantum_efficiency)\n")   # Write the data row for this specific run


    # write(f, ("k_eEPD_eGPD_gem = %1.3e/ps, dV_eEPD_eGPD = %1.3eV, distance_PD_PD     = ZERO!!", ode_p.k_eEPD_eGPD_gem, -dV_eEPD_eGPD))
    # write(f, ("k_eEPD_eGPD_bim = %1.3e/ps, dV_eEPD_eGPD = %1.3eV, distance_PD_D_bim  = %1.3eA", ode_p.k_eEPD_eGPD_bim, -dV_eEPD_eGPD, p.d_PD_D_bim))
    # write(f, ("k_eEPD_hD_bim  = %1.3e/ps, dV_eEPD_hD  = %1.3eV, distance_PD_D_bim  = %1.3eA", ode_p.k_eEPD_hD_bim,  -dV_eEPD_hD,  p.d_PD_D_bim))
    # write(f, ("k_eEPD_eA1_gem  = %1.3e/ps, dV_eEPD_eA1  = %1.3eV, distance_PD_eA1     = %1.3eA", ode_p.k_eEPD_eA1_gem,  -dV_eEPD-eA1,  d_edges_PD_A1))
    # write(f, ("k_eEPD_eA2_gem  = %1.3e/ps, dV_eEPD-eA2  = %1.3eV, distance_PD-eA2     = %1.3eA", ode_p.k_eEPD-eA2_gem,  -dV-eEPD-eA2,  d_edges_PD_A2))
    # write(f, ("k_D-eGPD_bim   = %1.3e/ps, dV_D-eGPD   = %1.3eV, distance_PD_D_bim  = %1.3eA", ode_p.k_D-eGPD_bim,   -dV_D-eGPD,   p.d_PD_D_bim))
    # write(f, ("k_eA1_eA2_gem   = %1.3e/ps, dV_eA1_eA2   = %1.3eV, distance_eA1_eA2     = %1.3eA", ode_p.k_eA1_eA2_gem,   -dV_eA1_eA2,   d_edges_A1_A2))
    # write(f, ("k_eA1_eEPD_gem  = %1.3e/ps, dV_eA1_eEPD  = %1.3eV, distance_PD_eA1     = %1.3eA", ode_p.k_eA1_eEPD_gem,  -dV_eA1_eEPD,  d_edges_PD_A1))
    # write(f, ("k_eA1_eGPD_gem  = %1.3e/ps, dV_eA1_eGPD  = %1.3eV, distance_PD_eA1     = %1.3eA", ode_p.k_eA1_eGPD_gem,  -dV_eA1_eGPD,  d_edges_PD_A1))
    # write(f, ("k_eA1_eGPD_bim  = %1.3e/ps, dV_eA1_eGPD  = %1.3eV, distance_PD_eA1     = %1.3eA", ode_p.k_eA1_eGPD_bim,  -dV_eA1_eGPD,  d_edges_PD_A1))
    # write(f, ("k_eA1_hD_bim   = %1.3e/ps, dV_eA1_hD   = %1.3eV, distance_eA1_D_bim  = %1.3eA", ode_p.k_eA1_hD_bim,   -dV_eA1_hD,   p.d_eA1_D_bim))
    # write(f, ("k_eA2_At_bim   = %1.3e/ps, dV_eA2_At   = %1.3eV, distance_eA2_At_bim = %1.3eA", ode_p.k_eA2_At_bim,   -dV_eA2_At,   p.d_eA2_At_bim))
    # write(f, ("k_eA2_eA1_gem   = %1.3e/ps, dV_eA2_eA1   = %1.3eV, distance_eA2_eA1     = %1.3eA", ode_p.k_eA2_eA1_gem,   -dV_eA2_eA1,   d_edges_A2_A1))
    # write(f, ("k_eA2_eGPD_gem  = %1.3e/ps, dV_eA2_eGPD  = %1.3eV, distance_PD_eA2     = %1.3eA", ode_p.k_eA2_eGPD_gem,  -dV_eA2_eGPD,  d_edges_PD_A2))
    # write(f, ("k_eA2_hD_bim   = %1.3e/ps, dV_eA2_hD   = %1.3eV, distance_eA2_D_bim  = %1.3eA", ode_p.k_eA2_hD_bim,   -dV_eA2_hD,   p.d_eA2_D_bim))



end





end #module






    # 








 

