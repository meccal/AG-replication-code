"""
LUCA MECCA
lmecca@london.edu
Simulate Aguiar, Gopinath (2006)
December 2022
"""

#INSTRUCTIONS: To allow for permanent/transitory shocks, you should change the version parameter
#in the AG_VFI_Discrete.jl file

using CSV, Distributions, Random, Interpolations, HPFilter

#Folder where the .jl file replicating the model is stored
const path="..."

const interpolation="No" #Choose "No" if for g_t and z_t you want to find the closest point of the grid, so effectively
#you don't need to interpolate on the policy functions
#Choose "Yes" if instead you want to allow values of g_t and z_t different from the grid points
################################################################
########################## PARAMETERS ##########################
################################################################
const K=500 #number of simulations
const T=10000 #number of quarters for each simulation
const brns=9500 #burn-in observations

################################################################


################################################################
########################## SIMULATION ##########################
################################################################
#First run the model
include(path * "/AG_VFI_Discrete.jl")

#Reshape the policy functions functions as matrices
D_t=permutedims(reshape(D_t, (n_a, n_z*n_g)), (2,1))
policy_asset=permutedims(reshape(policy_asset, (n_a, n_z*n_g)), (2,1))
q_t=permutedims(reshape(q_t, (n_a, n_z*n_g)), (2,1))


#The simulation can only be run for models which allow either the permanent or the transitory shock (not both)
if version == "complete"
    return error("The simulation can be run only for the model with transitory or permanent shocks")
end

#create containers for quantities of interest
std_y=Matrix{Any}(undef,K,1) #standard deviation of income
std_c=Matrix{Any}(undef,K,1) #standard deviation of consumption
std_TBy=Matrix{Any}(undef,K,1) #standard deviation of trade balance over income
std_R=Matrix{Any}(undef,K,1) #standard devtiation of the interest rate spread
corr_TBy_Y=Matrix{Any}(undef,K,1) #correlation of trade balance over income and income
corr_R_Y=Matrix{Any}(undef,K,1) #correlation of interest rate spread and income
corr_R_TBy=Matrix{Any}(undef,K,1) #correlation of interest rate spread and trade balance over income
corr_c_y=Matrix{Any}(undef,K,1) #correlation of cosumption and income
Default_number=Matrix{Any}(undef,K,1) #number of defaults
max_R=Matrix{Any}(undef,K,1) #maximum spread

#Simulate the paths
for k in 1:K
    if mod(k,10)==0
        print("Simulation number " * string(k)*"\n")
    end
    #Simulate the state variables z_t and g_t
    if version=="permanent"
        z_t=zeros(T+1,1) #z is constant in this case
        Random.seed!(1234+10*k)
        g_shocks=rand(Normal(), T+1) #draw standard Normally distributed shocks for G
        g_t=Matrix{Any}(undef,T+1,1)
        #initialize the process for g_t assimung g_0=μ_g
        
        if interpolation=="No"
            #Find the grid points which is closest to the actual realziation to avoid interpolation
            g_t[1]=g_grid[findmin(abs.(g_grid.-exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(μ_g) + σ_g*g_shocks[1])))[2]]
            for t in 2:T+1
                g_t[t]=g_grid[findmin(abs.(g_grid.-exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(g_t[t-1]) + σ_g*g_shocks[t])))[2]]
            end
        else #if we allow g_t to be different from the grid points
            #initialize the process for g_t assimung g_0=μ_g
            g_t[1]=exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(μ_g) + σ_g*g_shocks[1])
            for t in 2:T+1
                g_t[t]=exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(g_t[t-1]) + σ_g*g_shocks[t])
            end
            #Avoid extrapolation
            g_t[g_t.>maximum(g_grid)].=maximum(g_grid)
            g_t[g_t.<minimum(g_grid)].=minimum(g_grid)
        end
            
        state_t=g_t #state variable in this model is g_t

    else #if version is transitory
        g_t=μ_g.*ones(T+1,1) #trend is constant in this case
        Random.seed!(1234+10*k)
        z_shocks=rand(Normal(), T+1)
        z_t=Matrix{Any}(undef,T+1,1)
        #initialize the process for z_t assimung z_0=μ_z
        if interpolation=="No"
            #Find the grid points which is closest to the actual realziation to avoid interpolation
            z_t[1]=z_grid[findmin(abs.(z_grid.-((1-ρ_z)*μ_z + ρ_z*μ_z + σ_z*z_shocks[1])))[2]]
            for t in 2:T+1
                z_t[t]=z_grid[findmin(abs.(z_grid.-((1-ρ_z)*μ_z + ρ_z*z_t[t-1] + σ_z*z_shocks[t])))[2]]
            end
        else #if we allow z_t to be different from the grid points
            z_t[1]=(1-ρ_z)*μ_z + ρ_z*μ_z + σ_z*z_shocks[1]
            for t in 2:T+1
                z_t[t]=(1-ρ_z)*μ_z + ρ_z*z_t[t-1] + σ_z*z_shocks[t]
            end

            #Avoid extrapolation
            z_t[z_t.>maximum(z_grid)].=maximum(z_grid)
            z_t[z_t.<minimum(z_grid)].=minimum(z_grid)
        end

        state_t=z_t #state variable in this model is z_t
    end

    #Endowment path (detrended)
    y_t=exp.(z_t).*g_t./μ_g

    #create redemption vector, it is equal to 1 when there is redemption, 0 otherwise
    λ_t=zeros(T)
    λ_t[1:Int(λ*T)].=1
    λ_t=shuffle(λ_t) #shuffle in random order

    #ASSUMPTIONS: 
    #1. The agent initially has an asset position which is at the middle of the grid
    #2. The agent in period 0 did not default

    #Create matrices for the values we  are interested in recording
    a_t=Matrix{Any}(undef,T+1,1) #matrix of asset path
    a_t[1]=a_grid[Int(round(n_a/2))]
    Default_t=Matrix{Any}(undef,T,1) #matrix of state of default
    price_t=Matrix{Any}(undef,T,1) #matrix of bonds prices
    def_event_t=zeros(T) #matrix that has entry of 1 if default event happens

    #create 2-Dimensional interpolations functions 
    Pf_Default=linear_interpolation((vec(y_grid), [a_grid;]), D_t) #Default
    Pf_Bonds=linear_interpolation((vec(y_grid), [a_grid;]), policy_asset) #Bond emission
    Pf_Price=linear_interpolation((vec(y_grid), [a_grid;]), q_t) #Price of the bonds

    #Initialize (we need this step because here we are assuming that in period 0 the economy did not default)
    Default_t[1]=Pf_Default(state_t[1], a_t[1])

    if Default_t[1]==0 #No default
        a_t[2]=Pf_Bonds(state_t[1], a_t[1]) #bonds emission when there is no default
        price_t[1]=Pf_Price(state_t[1], a_t[2]) #price of the bond emission
    else #default
        def_event_t[1]=1
        a_t[2]=0 #bonds emission in case of default
        price_t[1]=(1+r)^(-1) #price of the bond emission
    end

    #complete iteration
    for t in 2:T
        if Default_t[t-1]==0 ||  Default_t[t-1]==1 && λ_t[t]==1 #did not default in the previous period or defaulted and is now readmitted
            #Decide whether to default or not
            Default_t[t]=Pf_Default(state_t[t], a_t[t])
        
            if Default_t[t]==0 #No default
                a_t[t+1]=Pf_Bonds(state_t[t], a_t[t]) #bonds emission when there is no default
                price_t[t]=Pf_Price(state_t[t], a_t[t+1]) #price of the bond emission
            else #default
                def_event_t[t]=1 #record a default
                a_t[t+1]=0 #bonds emission in case of default
                price_t[t]=(1+r)^(-1) #price of the bond emission
            end
        
        else #if the government defaulted in the previous period and wasn't redeemed
            Default_t[t]=1 #stay in default
            a_t[t+1]=0 #bonds emission in case of default
            price_t[t]=(1+r)^(-1) #price of the bond emission
        end
    end

    
    #take logs of the income process and bring back the trend
    log_y=log.(y_t)+cumsum(log.(g_t)[:,1]).+log(μ_g).-log.(g_t)
    #bring back the trend to bonds emissions
    a_t=a_t.*exp.(cumsum(log.(g_t)[:,1]).+log(μ_g).-log.(g_t))
    #trade balance
    TB_t=price_t.*a_t[2:T+1]-a_t[1:T]
    TB_t[Default_t.==1].=0 #when the country defaults does not repay bonds and does not issue new ones, i.e. its trade balance is zero
    #consumption
    c_t=exp.(log_y[1:T]).-TB_t
    c_t[Default_t.==1].=exp.(log_y[1:T,:])[Default_t.==1].*(1-δ) #when the country defaults, it faces a pensalty

    #interest rate spread
    Rs_t=(price_t).^(-1) .-1 .-r
  

    #Remove the trend component using the Hodrick-Prescott (HP) filter and drop burn in observations
    logy_t_cycle = log_y[brns+1:T,1]-HP(log_y[brns+1:T,1], 1600) #income(logs)
    logc_t_cycle= log.(c_t[brns+1:T])-HP(log.(c_t[brns+1:T,1]), 1600) #consumption (logs)
    TBy_t_cycle=TB_t[brns+1:T]./exp.(log_y[brns+1:T])-HP(TB_t[brns+1:T]./exp.(log_y[brns+1:T]), 1600) #trade balance over output
    Rs_t_cycle=Rs_t[brns+1:T]-HP(Rs_t[brns+1:T], 1600) #spread

    #compute statistics
    std_y[k]=std(logy_t_cycle)*100 
    std_c[k]=std(logc_t_cycle)*100
    std_TBy[k]=std(TBy_t_cycle)*100
    std_R[k]=std(Rs_t_cycle)*100
    corr_c_y[k]=cor(logy_t_cycle, logc_t_cycle)
    corr_TBy_Y[k]=cor(TBy_t_cycle, logy_t_cycle)[1,1]
    corr_R_Y[k]=cor(Rs_t_cycle, logy_t_cycle)[1,1]
    corr_R_TBy[k]=cor(Rs_t_cycle, TBy_t_cycle)[1,1]
    Default_number[k]=mean(def_event_t[brns+1:T,1])
    max_R[k]=maximum(Rs_t_cycle)*10000 #basis points
end

#we are annualizing the standard deviation of the spread and the maximum spread.
#Reproduce Table 3 of Aguiar-Gopinath (2006)
col=[mean(std_y), mean(std_c), mean(std_TBy), mean(std_R)*4, mean(corr_c_y), mean(corr_TBy_Y),
mean(corr_R_Y), mean(corr_R_TBy), mean(Default_number)*10000, mean(max_R)*4]







