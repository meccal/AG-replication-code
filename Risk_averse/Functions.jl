"""
LUCA MECCA
lmecca@london.edu
September 2022
"""
#This file contains the functions that execute performance sensitive parts of the code
#These functions are more specific to this project

#Computes joint probability of movements in the state variables g,z
function joint_prob(T_matrix_x::Matrix{Float64}, T_matrix_z::Union{Matrix{Float64}, Vector{Int64}}, T_matrix_g::Union{Matrix{Float64}, Vector{Int64}}, n_x::Int64, n_z::Int64, n_g::Int64)::Matrix{Any}
    T_matrix_g_z_x=Matrix{Any}(undef,n_g*n_z*n_x,1)
    for i in 1:n_g, j=1:n_z, k in 1:n_x
        T_matrix_g_z_x[k+(i-1)*n_z*n_x+(j-1)*n_x]=repeat(T_matrix_x[:,k], outer=n_g*n_z).*repeat(repeat(T_matrix_z[:,j], inner=n_x), outer=n_g).*
        repeat(T_matrix_g[:,i], inner=n_x*n_z)
    end
    return T_matrix_g_z_x
end

#Finds the fixed point of the price to consumption ratio and pins down the SDF
function SDF(T_matrix_x::Matrix{Float64}, x_grid::Matrix{Float64}, β_L::Union{Float64, Int64}, γ_L::Union{Int64, Float64}, θ::Float64, μ_c::Float64, σ_c::Float64, n_x::Int64)::Matrix{Float64}
    #Initialize the price to consumption ratio
    global PC_t=zeros(n_x,1)
    #Pin down the SDF by find ind the fixed point for PC_t
    #Policy function iteration continues until difference is lower than 10^(-6) or the number of iterations > 10,000
    global difference=1
    global counter=0
    global error_list_PC=ones(0)
    #constant part of the iteration
    constant=β_L^θ.*exp.((1-γ_L).*(μ_c.+x_grid).+0.5*(1-γ_L)^2*σ_c^2)
    while difference>10^(-6) && counter<10000
        global counter+=1
        if mod(counter,20)==0
            print("Iteration number " * string(counter)* " for the SDF\n")
        end

        #expected value part
        exp_value=[sum(T_matrix_x[:,j].*(1 .+PC_t).^θ) for j in 1:n_x]

        
        global PC_t1=(constant .* exp_value).^(1/θ)
        #Now compute the difference between the newly found value function and the previous one
        global difference=sum(abs.(PC_t-PC_t1))
        append!(error_list_PC, difference)
        #If the difference is too large, set Vf=Vf_1 and restart:
        global PC_t=PC_t1

    end
    return PC_t
end

#Computes the utility function when the government does not default
function U_fun(a_grid:: LinRange{Float64, Int64}, z_grid::Union{Vector{Int64},Matrix{Float64},Vector{Float64}}, g_grid::Union{Vector{Int64}, Matrix{Float64}, LinRange{Float64, Int64}, Vector{Float64}}, q_t::Matrix{Float64}, n_a::Int64, n_x::Int64, n_z::Int64, n_g::Int64)::Vector{Float64}
    U=((exp.(repeat(repeat(z_grid, inner=n_a*n_a*n_x), outer=n_g)).*
    repeat(g_grid, inner=n_a*n_z*n_a*n_x)./μ_g.+
    repeat(repeat(a_grid, inner=n_a), outer=n_g*n_z*n_x).-
    reshape(convert(Array,VectorOfArray([repeat(q_t[1+(i-1)*n_a:n_a+(i-1)*n_a], outer=n_a) for i in 1:n_g*n_z*n_x])), n_a*n_a*n_z*n_g*n_x).*
    repeat(a_grid, outer=n_g*n_a*n_z*n_x).*
    repeat(g_grid, inner=n_a*n_z*n_a*n_x)).^(1-γ))./(1-γ)

    return U
end

#Computes the continuation value when the government does not default
function EV_fun(V_t::Matrix{Float64}, T_matrix_g_z_x::Matrix{Any}, n_a::Int64, n_x::Int64, n_z::Int64, n_g::Int64)::Vector{Float64}
    EV=Matrix{Float64}(undef,n_g*n_z*n_x*n_a,1)
    V_t_matrix=permutedims(reshape(V_t, (n_a, n_z*n_g*n_x)), (2,1)) #useful for next step
    for i in 1:n_g, j in 1:n_z, k in 1:n_x
        EV[(i-1)*n_z*n_a*n_x+(j-1)*n_a*n_x+(k-1)*n_a+1:(i-1)*n_z*n_a*n_x+(j-1)*n_a*n_x+(k-1)*n_a+n_a]=sum(V_t_matrix.*repeat(T_matrix_g_z_x[k+(j-1)*n_x+(i-1)*n_x*n_z], 1, n_a), dims=1)
    end
    global EV=reshape(convert(Array,VectorOfArray([repeat(EV[1+(i-1)*n_a:n_a+(i-1)*n_a], outer=n_a) for i in 1:n_g*n_z*n_x])), n_a*n_a*n_z*n_g*n_x)
    return EV
end

#Given the utility function and continuation value, updates the value function in case of good credit history
function VG(U::Vector{Float64}, EV::Vector{Float64}, β::Float64, n_a::Int64, n_x::Int64, n_z::Int64, n_g::Int64)::Matrix{Float64}
    VG_t1=findmax(permutedims(reshape(U.+β.*EV, (n_a, n_g*n_z*n_a*n_x)), (2,1)), dims=2)[1]
    return VG_t1
end

#Computes consumption when the government is in autarky
function C_bad(z_grid::Union{Vector{Int64},Matrix{Float64}, Vector{Float64}}, g_grid::Union{Vector{Int64}, Matrix{Float64}, LinRange{Float64, Int64}, Vector{Float64}}, δ::Float64, μ_g::Union{Float64, Int64})::Vector{Float64}
    C=(1-δ).*exp.(repeat(z_grid, outer=n_g)).*repeat(g_grid, inner=n_z)./μ_g
    return C
end

#Computes the continuation values when the government is in autarky
function EV_bad(V_t::Matrix{Float64},VB_t::Matrix{Float64}, T_matrix_g_z_x::Matrix{Any}, n_a::Int64, n_x::Int64, n_z::Int64, n_g::Int64)
    EV0=Matrix{Float64}(undef,n_g*n_z*n_x,1)
    EVB=Matrix{Float64}(undef,n_g*n_z*n_x,1)

    for i in 1:n_g, j in 1:n_z, k in 1:n_x
        EV0[k+(j-1)*n_x+(i-1)*n_x*n_z]=sum(T_matrix_g_z_x[k+(j-1)*n_x+(i-1)*n_x*n_z].*V_t[n_a:n_a:n_a*n_g*n_z*n_x]) #pick values of V that correspond to value of the asset = 0 (re-enter the financial markets)
        EVB[k+(j-1)*n_x+(i-1)*n_x*n_z]=sum(T_matrix_g_z_x[k+(j-1)*n_x+(i-1)*n_x*n_z].*VB_t) 
    end

    return EV0, EVB
end

#Updates the value function in case of good credit history
function VB(C::Vector{Float64}, EV0::Matrix{Float64}, EVB::Matrix{Float64}, γ::Int64, λ::Float64, β::Float64, n_x::Int64)::Matrix{Float64}
    VB_t1=repeat(C, inner=n_x).^(1-γ)./(1-γ)+λ.*β.*EV0.+(1-λ).*β.*EVB
    return VB_t1
end

#Updates the price of debt q_t
function q(D_t::Matrix{Int64},T_matrix_g_z_x::Matrix{Any}, n_a::Int64, n_x::Int64, n_z::Int64, n_g::Int64, r::Float64)::Matrix{Float64}
    D_t_matrix=permutedims(reshape(D_t, (n_a, n_z*n_g*n_x)), (2,1)) #useful for next step
    #And we can update the pricing of the debt
    q_t1=Matrix{Float64}(undef,n_g*n_z*n_x*n_a,1)
    for i in 1:n_g, j in 1:n_z, k in 1:n_x
        q_t1[(i-1)*n_z*n_a*n_x+(j-1)*n_a*n_x+(k-1)*n_a+1:(i-1)*n_z*n_a*n_x+(j-1)*n_a*n_x+(k-1)*n_a+n_a]=
        (β_L^θ*exp(-γ_L*(μ_c+x_grid[k])+ 0.5*γ_L^2*σ_c^2)*PC_t[k]^(1-θ).*
        sum(repeat(T_matrix_g_z_x[k+(j-1)*n_x+(i-1)*n_x*n_z], 1, n_a).* repeat(repeat((1 .+PC_t).^(θ-1), outer=n_g*n_z), 1, n_a) .* (1 .-D_t_matrix), dims=1))./(1+r)
    end
    return q_t1
end 


