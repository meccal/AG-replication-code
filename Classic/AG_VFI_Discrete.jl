"""
LUCA MECCA
lmecca@london.edu
Replicate the results of Aguiar, Gopinath (2006) using value function iteration (VFI) and Tauchen discretization
December 2022
Written and tested in Julia 1.8
"""

using Distributions, Plots, RecursiveArrayTools
include("Discretize.jl")
include("Functions.jl")


#################################################################
########################## CALIBRATION ##########################
#################################################################
const version="transitory" #choose "permanent" if you want to run the version of the model with only shocks to the trend
#choose "transitory" to run the version of the model with only transitory shocks.
#choose "complete" if you want to allow for both transitory and permanent shocks

const g_grid_version="logs" #choose "logs" if you want to have an equally-spaced grid in logs
#choose "levels" if you want to have an equally spaced grid in levels (as in the paper)

#Take the parameters for the quartely AG06 calibration
const γ=2 #government's risk aversion
const r=0.01 #world interest rate
const β=0.8 #discount factor

const δ=0.02 #loss of output in autarky
const λ=0.1 #probability of redemption

const μ_g=1.006 #unconditional mean of the trend shock
const σ_g=0.03 #standard deviation of the trend shock
const ρ_g=0.17 #autocorrelation coefficient of the trend shock
const w_g=4.1458 #controls for the span of the grid

const σ_z=0.034 #standard deviation of the transitory shock
const μ_z=-0.5*σ_z^2 #unconditional mean of the transitory shock
const ρ_z=0.9 #autocorrelation coefficient of the transitory shock
const w_z=2.5 #controls for the span of the grid

const n_a=400 #number of grid points for the asset (a)

if version == "permanent"
    n_g=25 #number of grid points for g_t
    n_z=1 #number of grid points for z_t
    a_min=-0.22 #maximum amount of debt
    a_max=0 #minimum amount of debt
elseif version =="transitory"
    const n_z=25 #number of grid points for z_t
    const n_g=1 #number of grid points for g_t
    const a_min=-0.3 #maximum amount of debt
    const a_max=0 #minimum amount of debt
elseif version == "complete"
    const n_z=25 #number of grid points for z_t
    const n_g=25 #number of grid points for g_t
    const a_min=-0.3 #maximum amount of debt
    const a_max=0 #minimum amount of debt
else
    return(error("version should be equal to complete, permanent or transitory."))
end
#################################################################


#################################################################
######################### DISCRETIZATION ########################
#################################################################
#We discretize the state variables: 
#a amount of debt (control/state - endogenous)
#g_t the trend shock 
#z_t the transitory shocks (state - exogenous)
@time begin #time recording
    #Endowment (y)
    if version=="transitory"
        z_grid, T_matrix_z=Tau(ρ_z, σ_z, n_z, w_z, μ_z*(1-ρ_z)) #grid (logs) and transition matrix for transitory shock z
        g_grid=[μ_g] #no shock (levels), so g_t is equal to μ_g
        T_matrix_g=[1] #no shock
    elseif version =="permanent"
        if g_grid_version=="logs"
            g_grid, T_matrix_g=Tau(ρ_g, σ_g, n_g, w_g) #grid (logs) and transition matrix for permanent shock g
            g_grid=exp.(g_grid.+log(μ_g)) #convert into levels
        elseif g_grid_version=="levels" #next section is taken from Aguiar and Gopinath (2006) code and translated
            c=σ_g/(1-ρ_g^2)^(.5) #stdev of invariant distribution of log g
            std_lev=(exp(2*(log(μ_g)-0.5*c^2)+2*c^2)-exp(2*(log(μ_g)-0.5*c^2)+c^2))^(0.5) #convert std in levels
            step_g=2*w_g/(n_g-1)*std_lev #space between each point of the grid
            g_grid=LinRange(μ_g-w_g*std_lev, μ_g+w_g*std_lev, n_g)

            #now compute the transition matrix
            T_matrix_g=Matrix{Float64}(undef,n_g,n_g)
            for i in 1:n_g, j in 1:n_g
                T_matrix_g[j,i]=cdf(LogNormal((1-ρ_g)*(log(μ_g)-0.5*c^2)+ρ_g*log(g_grid[i]), σ_g), g_grid[j]+step_g/2)-cdf(LogNormal((1-ρ_g)*(log(μ_g)-0.5*c^2)+ρ_g*log(g_grid[i]), σ_g), g_grid[j]-step_g/2)
            end

        else
            return(error("The version of the grid for g must be either logs or levels"))
        end
        z_grid=[0]
        T_matrix_z=[1] #no shock
    elseif version == "complete"
        z_grid, T_matrix_z=Tau(ρ_z, σ_z, n_z, w_z) #grid (logs) and transition matrix for transitory shock z
        z_grid=z_grid.+μ_z
        g_grid, T_matrix_g=Tau(ρ_g, σ_g, n_g, w_g) #grid (logs) and transition matrix for permanent shock g
        g_grid=exp.(g_grid.+log(μ_g))
    else 
    end

    #Debt (a)
    const a_grid=LinRange(a_min, a_max, n_a)
    #convert grids into vectors
    g_grid=vec(g_grid)
    z_grid=vec(z_grid)

    #Now compute the transtion matrices that include the probabilities of contemporaneous changes in both g_t and z_t
    #for each (g_t, z_t) pair we create a vector that includes the proability of moving to the pair (g_{t+1}, z_{t+1})
    #Order is relevant, first we have all possible values for z for the same value of g, and so on
    T_matrix_g_z=joint_prob(T_matrix_z, T_matrix_g, n_z, n_g)

    #################################################################


    #################################################################
    ########################### ITERATION ###########################
    #################################################################
    #ORDER IS RELEVANT: the lst variable in the parenthesis is the one that moves first in the vector and so on
    #Initialize price function q(z_t, g_t, a_{t+1}) is a (n_g*n_z*n_a,1) vector
    #Initialized at the risk-free rate (assuming that probability of default is zero)
    q_t=ones(n_g*n_z*n_a,1).*((1+r)^(-1))

    #Initialize the value functions
    #V^G(a_t, z_t, g_t), the value function if in good credit state, is a (n_g*n_z*n_a,1) vector
    VG_t=zeros(n_g*n_z*n_a,1)
    #V^B(z_t, g_t), the value function if in bad credit state is a (n_g*n_z) vector (does not depend on debt)
    VB_t=zeros(n_g*n_z, 1)
    #V(a_t, z_t, g_t)=max(V^G(a_t, z_t, g_t), V^B(g_t, z_t))
    V_t=zeros(n_g*n_z*n_a,1)

    #Value function iteration continues until difference is lower than 10^(-6) or the number of iterations > 10,000
    #i is present amount of bonds a_t
    #h is future amount of bonds a_{t+1}
    difference=1
    counter=0
    error_list=ones(0)

    while difference>10^(-6) && counter<10000
        global counter+=1
        if mod(counter,20)==0
            print("Iteration number " * string(counter)*"\n")
        end

        #GOOD STATE
        #Start with the value function if the credit history is good (G)
        #Compute the amount of utility (consumption) for each possible level of the state variables (a_t, g_t, z_t) and choice variable (a_{t+1})
        #order is (g_t, z_t, a_t, a_{t+1})
        global U=U_fun(a_grid, z_grid, g_grid, q_t, n_a, n_z, n_g)
        #Expected continuation value
        global EV=EV_fun(V_t, T_matrix_g_z, n_a, n_z, n_g)

        #take the maximum achievable value
        VG_t1=VG(U, EV, β, n_a, n_z, n_g)

        #BAD STATE
        #Now compute the value function if the credit history is bad
        #amount of consumption in case of exclusion from the financial markets (with penalty δ)
        C=C_bad(z_grid, g_grid, δ, μ_g)

        #Compute:
        #EV0: Continuation value if the country is admissed back in financial markets (with zero debt)
        #EVB: Continuation value if the country is not admitted back in financial markets 
        EV0, EVB=EV_bad(V_t, VB_t, T_matrix_g_z, n_a, n_z, n_g)
    
        #Update the value function
        VB_t1=VB(C, EV0, EVB, γ, λ, β)

        #Update the value function V_t by choosing the max between V_B and VG_t
        V_t1=(VG_t1.>repeat(vec(VB_t1), inner=n_a)).*VG_t1
        V_t1[V_t1.==0].=repeat(vec(VB_t1), inner=n_a)[:,:][V_t1.==0]

        #DEFAULT DECISION
        #Now we can derive the default decision (when value of good credit state is lower than bad credit state)
        global D_t=(VG_t1.<repeat(vec(VB_t1), inner=n_a)).*1
        #Update the price of debt
        q_t1=q(D_t, T_matrix_g_z, n_a, n_z, n_g, r)

        #Difference
        global difference=max(sum(abs.(VB_t1-VB_t)), sum(abs.(VG_t1-VG_t)))
        append!(error_list, difference)
        #If difference is not small enough, keep iterating
        global VB_t=VB_t1
        global VG_t=VG_t1
        global V_t=V_t1
        global q_t=q_t1
    end #while loop
end #time

#Find policy function for assets, i.e. the optimal amount of bonds issued when the country decides not to default
policy_asset=repeat(a_grid', outer=n_g*n_z*n_a)[findmax(permutedims(reshape(U.+β.*EV, (n_a, n_g*n_z*n_a)), (2,1)), dims=2)[2]]
#################################################################




#################################################################
############################# PLOTS #############################
#################################################################
#Replicate Figure 2 of the paper
#Setups for charts are based on whether the version is transitory or permanent (to replicate the Figures of the paper)
if version=="permanent"
    n_y=n_g
    y_grid=g_grid
    lab="g"
    xmin=-0.25
    xmax=-0.18

elseif version=="transitory"
    n_y=n_z
    y_grid=z_grid
    lab="z"
    xmin=-0.3
    xmax=-0.23
else
    return(error("To produce this chart, version should be either permanent or temporary"))
end



fig = plot(heatmap(permutedims(reshape(D_t, (n_a, n_z*n_g)), (2,1))), xticks=(1:n_a/4:(n_a+1),[string(a_grid[1]), string(round(a_grid[Int(n_a/4)], digits=2)), string(round(a_grid[Int(n_a/2)], digits=2)), string(round(a_grid[Int(n_a*3/4)], digits=2)), string(a_grid[n_a])]),
yticks=(1:n_y/5:n_y+1, [string(round(y_grid[1], digits=2)), string(round(y_grid[Int(n_y/5)], digits=2)), string(round(y_grid[Int(n_y/5*2)], digits=2)), string(round(y_grid[Int(n_y*3/5)], digits=2)), string(round(y_grid[Int(n_y*4/5)], digits=2)), string(round(y_grid[n_y], digits=2)) ]), 
colorbar=false, xlabel="Assets", ylabel=lab)
#savefig("heatmap_" *version*".png")

#Replicate Figure 3 of the paper
plot(a_grid, q_t[1:n_a], xlabel="Assets", ylabel="Price of the Bond",line=(:dash, 2), color=:black, xlim=(xmin, xmax), ylim=(0,1), label="q(a, min(" *lab* "))") #price for each level of asset corresponding to the lowest level of permanent/transitory shock
plot!(a_grid, q_t[lastindex(q_t)-n_a+1:lastindex(q_t)], xlabel="Assets", ylabel="Price of the Bond",color=:black, label="q(a, max("*lab*"))") #price for each level of asset corresponding to the highest level of permanent/transitory shock
plot!(legend=:bottomright)
#savefig("price_" *version*".png")



