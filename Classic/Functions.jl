"""
LUCA MECCA
lmecca@london.edu
September 2022
"""
#This file contains the functions that execute performance sensitive parts of the code
#These functions are more specific to this project

#Computes joint probability of movements in the state variables g,z
function joint_prob(T_matrix_z::Union{Matrix{Float64}, Vector{Int64}}, T_matrix_g::Union{Matrix{Float64}, Vector{Int64}}, n_z::Int64, n_g::Int64)
    T_matrix_g_z=Matrix{Any}(undef,n_g,n_z)
    for i in 1:n_g, j=1:n_z
        T_matrix_g_z[i,j]=repeat(T_matrix_g[:,i], inner=n_z).*repeat(T_matrix_z[:,j], outer=n_g)
    end
    return T_matrix_g_z
end


#Computes the utility function when the government does not default
function U_fun(a_grid:: LinRange{Float64, Int64}, z_grid::Union{Vector{Int64},Matrix{Float64},Vector{Float64}}, g_grid::Union{Vector{Int64}, Matrix{Float64}, LinRange{Float64, Int64}, Vector{Float64}}, q_t::Matrix{Float64}, n_a::Int64, n_z::Int64, n_g::Int64)
    U=((exp.(repeat(repeat(z_grid, inner=n_a*n_a), outer=n_g)).*
    repeat(g_grid, inner=n_a*n_z*n_a)./μ_g.+
    repeat(repeat(a_grid, inner=n_a), outer=n_g*n_z).-
    reshape(convert(Array,VectorOfArray([repeat(q_t[1+(i-1)*n_a:n_a+(i-1)*n_a], outer=n_a) for i in 1:n_g*n_z])), n_a*n_a*n_z*n_g).*
    repeat(a_grid, outer=n_g*n_a*n_z).*
    repeat(g_grid, inner=n_a*n_z*n_a)).^(1-γ))./(1-γ)

    return U
end

#Computes the continuation value when the government does not default
function EV_fun(V_t::Matrix{Float64}, T_matrix_g_z::Matrix{Any}, n_a::Int64, n_z::Int64, n_g::Int64)
    EV=Matrix{Float64}(undef,n_g*n_z*n_a,1)
    V_t_matrix=permutedims(reshape(V_t, (n_a, n_z*n_g)), (2,1)) #useful for next step
    for i in 1:n_g, j in 1:n_z
        EV[(i-1)*n_a*n_z+(j-1)*n_a+1:(i-1)*n_a*n_z+n_a+(j-1)*n_a]=sum(V_t_matrix.*repeat(T_matrix_g_z[i,j], 1, n_a), dims=1)
    end
    EV=reshape(convert(Array,VectorOfArray([repeat(EV[1+(i-1)*n_a:n_a+(i-1)*n_a], outer=n_a) for i in 1:n_g*n_z])), n_a*n_a*n_z*n_g)
    return EV
end

#Given the utility function and continuation value, updates the value function in case of good credit history
function VG(U::Vector{Float64}, EV::Vector{Float64}, β::Float64, n_a::Int64, n_z::Int64, n_g::Int64)
    VG_t1=findmax(permutedims(reshape(U.+β.*EV, (n_a, n_g*n_z*n_a)), (2,1)), dims=2)[1]
    return VG_t1
end

#Computes consumption when the government is in autarky
function C_bad(z_grid::Union{Vector{Int64},Matrix{Float64}, Vector{Float64}}, g_grid::Union{Vector{Int64}, Matrix{Float64}, LinRange{Float64, Int64}, Vector{Float64}}, δ::Float64, μ_g::Union{Float64, Int64})
    C=(1-δ).*exp.(repeat(z_grid, outer=n_g)).*repeat(g_grid, inner=n_z)./μ_g
    return C
end

#Computes the continuation values when the government is in autarky
function EV_bad(V_t::Matrix{Float64},VB_t::Matrix{Float64}, T_matrix_g_z::Matrix{Any}, n_a::Int64, n_z::Int64, n_g::Int64)
    EV0=Matrix{Float64}(undef,n_g*n_z,1)
    EVB=Matrix{Float64}(undef,n_g*n_z,1)

    for i in 1:n_g, j in 1:n_z
        EV0[(i-1)*n_z+j]=sum(T_matrix_g_z[i,j].*V_t[n_a:n_a:n_a*n_g*n_z]) #pick values of V that correspond to value of the asset = 0 (re-enter the financial markets)
        EVB[(i-1)*n_z+j]=sum(T_matrix_g_z[i,j].*VB_t) 
    end

    return EV0, EVB
end

#Updates the value function in case of good credit history
function VB(C::Vector{Float64}, EV0::Matrix{Float64}, EVB::Matrix{Float64}, γ::Int64, λ::Float64, β::Float64)
    VB_t1=C.^(1-γ)./(1-γ)+λ.*β.*EV0.+(1-λ).*β.*EVB
    return VB_t1
end

#Updates the price of debt q_t
function q(D_t::Matrix{Int64},T_matrix_g_z::Matrix{Any}, n_a::Int64, n_z::Int64, n_g::Int64, r::Float64)
    D_t_matrix=permutedims(reshape(D_t, (n_a, n_z*n_g)), (2,1)) #useful for next step
    q_t1=Matrix{Float64}(undef,n_g*n_z*n_a,1)
    for i in 1:n_g, j in 1:n_z
        q_t1[(i-1)*n_a*n_z+(j-1)*n_a+1:(i-1)*n_a*n_z+n_a+(j-1)*n_a]=(1 .-sum(repeat(T_matrix_g_z[i,j], 1, n_a).*D_t_matrix, dims=1))./(1+r)
    end
    return q_t1
end 