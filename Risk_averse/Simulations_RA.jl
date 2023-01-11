"""
LUCA MECCA
lmecca@london.edu
Simulate Aguiar, Gopinath (2006)
December 2022
"""

#INSTRUCTIONS: To allow for permanent/transitory shocks, you should change the version parameter
#in the AG_VFI_RA.jl file

using CSV, Distributions, Random, Interpolations, HPFilter

#Folder where the .jl file replicating the model is stored
const path="C:/Users/lmecca/OneDrive - London Business School/Research/Replications/Aguiar-Gopinath/AG_replication_code/Risk_averse"
const report="mean" #Choose "mean if you want to report the mean of the simulation paths, choose "median" otherwise
const interpolation="Yes" #Choose "No" if for g_t and z_t you want to find the closest point of the grid, so effectively
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
############################# SETUP ############################
################################################################
#First run the model
include(path * "/AG_VFI_RA.jl")

if version =="complete"
    #Reshape the policy functions functions as 4D matrices
    #First dimension is x, second is a, third is z, and fourth is g.
    D_t_reshape=permutedims(reshape(D_t, (n_a, n_x, n_z, n_g)), (2,1,3,4))
    policy_asset_reshape=permutedims(reshape(policy_asset, (n_a, n_x, n_z, n_g)), (2,1,3,4))
    q_t_reshape=permutedims(reshape(q_t, (n_a, n_x, n_z, n_g)), (2,1,3,4))
else #if version is "transitory" or "complete"
    #Reshape the policy functions functions as 4D matrices
    #First dimension is x, second is a, third is y
    #where y=g if version is "permanent" and y=g if version is "transitory"
    #Recall n_y=n_g if version is "permanent" and n_y=n_z if version is "transitory"
    D_t_reshape=permutedims(reshape(D_t, (n_a, n_x, n_y)), (2,1,3))
    policy_asset_reshape=permutedims(reshape(policy_asset, (n_a, n_x, n_y)), (2,1,3))
    q_t_reshape=permutedims(reshape(q_t, (n_a, n_x, n_y)), (2,1,3))
end


#create containers for quantities of interest
std_y=Matrix{Float64}(undef,K,1) #standard deviation of income
std_c=Matrix{Float64}(undef,K,1) #standard deviation of consumption
std_TBy=Matrix{Float64}(undef,K,1) #standard deviation of trade balance over income
std_R=Matrix{Float64}(undef,K,1) #standard devtiation of the interest rate spread
corr_TBy_Y=Matrix{Float64}(undef,K,1) #correlation of trade balance over income and income
corr_R_Y=Matrix{Float64}(undef,K,1) #correlation of interest rate spread and income
corr_R_TBy=Matrix{Float64}(undef,K,1) #correlation of interest rate spread and trade balance over income
corr_c_y=Matrix{Float64}(undef,K,1) #correlation of cosumption and income
Default_number=Matrix{Float64}(undef,K,1) #number of defaults
max_R=Matrix{Float64}(undef,K,1) #maximum spread

############################################################################################################################
#Interpolation Functions
#crate an interpolation function to get the index to use in subsequent interpolation
idx_fun_x=linear_interpolation(vec(x_grid), 1:n_x)
idx_fun_a=linear_interpolation(a_grid, 1:n_a)
if version=="complete"
    idx_fun_z=linear_interpolation(vec(z_grid), 1:n_z)
    idx_fun_g=linear_interpolation(vec(g_grid), 1:n_g)
else
    idx_fun_state=linear_interpolation(vec(y_grid), 1:n_y)
end

#create interpolation function
Pf_Default=interpolate(D_t_reshape,BSpline(Linear()))
Pf_Bonds=interpolate(policy_asset_reshape,BSpline(Linear()))
Pf_Price=interpolate(q_t_reshape,BSpline(Linear()))


################################################################
########################## SIMULATION ##########################
################################################################
#Simulate the paths
for k in 1:K
    if mod(k,10)==0
        print("Simulation number " * string(k)*"\n")
    end
    #Simulate the state variables z_t and g_t and x_t
    Random.seed!(1234+10*k)
    x_shocks=rand(Normal(), T+1) #draw standard Normally distributed shocks for x
    Random.seed!(12345+100*k)
    z_shocks=rand(Normal(), T+1) #draw standard Normally distributed shocks for z
    Random.seed!(123456+1000*k)
    g_shocks=rand(Normal(), T+1) #draw standard Normally distributed shocks for g
    
    x_t=Matrix{Float64}(undef,T+1,1)
    z_t=Matrix{Float64}(undef,T+1,1)
    g_t=Matrix{Float64}(undef,T+1,1)
    
    if interpolation=="No"
        #Find the grid points which is closest to the actual realziation to avoid interpolation
        x_t[1]=x_grid[findmin(abs.(x_grid.-(ρ_c*0+ϕ_e*σ_c*x_shocks[1])))[2]]
        z_t[1]=z_grid[findmin(abs.(z_grid.-((1-ρ_z)*μ_z + ρ_z*μ_z + σ_z*z_shocks[1])))[2]]
        g_t[1]=g_grid[findmin(abs.(g_grid.-exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(μ_g) + σ_g*g_shocks[1])))[2]]
        for t in 2:T+1
            x_t[t]=x_grid[findmin(abs.(x_grid.-(ρ_c*x_t[t-1]+ϕ_e*σ_c*x_shocks[t])))[2]]
            z_t[t]=z_grid[findmin(abs.(z_grid.-((1-ρ_z)*μ_z + ρ_z*z_t[t-1] + σ_z*z_shocks[t])))[2]]
            g_t[t]=g_grid[findmin(abs.(g_grid.-exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(g_t[t-1]) + σ_g*g_shocks[t])))[2]]
        end
    else #if we allow g_t to be different from the grid points
        #intialize the process for x_t assuming x_0=0
        x_t[1]=ρ_c*0+ϕ_e*σ_c*x_shocks[1]
        #initialize the process for z_t assumin z_0=μ_z
        z_t[1]=(1-ρ_z)*μ_z + ρ_z*μ_z + σ_z*z_shocks[1]
        #initialize the process for g_t assimung g_0=μ_g
        g_t[1]=exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(μ_g) + σ_g*g_shocks[1])
        for t in 2:T+1
            x_t[t]=ρ_c*x_t[t-1]+ϕ_e*σ_c*x_shocks[t]
            z_t[t]=(1-ρ_z)*μ_z + ρ_z*z_t[t-1] + σ_z*z_shocks[t]
            g_t[t]=exp((1-ρ_g)*(log(μ_g)-0.5*σ_g^2/(1-ρ_g^2)) + ρ_g*log(g_t[t-1]) + σ_g*g_shocks[t])
        end
        #Avoid extrapolation
        g_t[g_t.>maximum(g_grid)].=maximum(g_grid)
        g_t[g_t.<minimum(g_grid)].=minimum(g_grid)
        x_t[x_t.>maximum(x_grid)].=maximum(x_grid)
        x_t[x_t.<minimum(x_grid)].=minimum(x_grid)
        z_t[z_t.>maximum(z_grid)].=maximum(z_grid)
        z_t[z_t.<minimum(z_grid)].=minimum(z_grid)
    end
    
    #adjuste the series based on the model
    if version=="permanent"
        z_t=zeros(T+1,1) #z is constant in this case
        state_t=g_t #state variable is g_t in this case
    elseif version=="transitory"
        g_t=μ_g.*ones(T+1,1) #trend is constant in this case
        state_t=z_t #state variable is z_t in this case
    else #version is "complete", both shocks are working in the model
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
    a_t=Matrix{Float64}(undef,T+1,1) #matrix of asset path
    a_t[1]=a_grid[Int(round(n_a/2))]
    Default_t=Matrix{Float64}(undef,T,1) #matrix of state of default
    price_t=Matrix{Float64}(undef,T,1) #matrix of bonds prices
    def_event_t=zeros(T) #matrix that has entry of 1 if default event happens

    #Initialize (we need this step because here we are assuming that in period 0 the economy did not default)
    #This cumbersome expression for default is due to the fact that (in very rare cases) interpolation could be between a 0 and a 1
    #We round to the closest between 0 and 1.
    if version=="complete"
        Default_t[1]=[0 1][findmin([abs(0-Pf_Default(idx_fun_x(x_t[1]), idx_fun_a(a_t[1]), idx_fun_z(z_t[1]), idx_fun_g(g_t[1]))) abs(1-Pf_Default(idx_fun_x(x_t[1]), idx_fun_a(a_t[1]), idx_fun_z(z_t[1]), idx_fun_g(g_t[1])))])[2]]
    else
        Default_t[1]=[0 1][findmin([abs(0-Pf_Default(idx_fun_x(x_t[1]), idx_fun_a(a_t[1]), idx_fun_state(state_t[1]))) abs(1-Pf_Default(idx_fun_x(x_t[1]), idx_fun_a(a_t[1]), idx_fun_state(state_t[1])))])[2]]
    end


    if Default_t[1]==0 #No default
        if version=="complete"
            a_t[2]=Pf_Bonds(idx_fun_x(x_t[1]), idx_fun_a(a_t[1]), idx_fun_z(z_t[1]), idx_fun_g(g_t[1])) #bonds emission when there is no default
            price_t[1]=Pf_Price(idx_fun_x(x_t[1]), idx_fun_a(a_t[2]), idx_fun_z(z_t[1]), idx_fun_g(g_t[1])) #price of the bond emission          
        else
            a_t[2]=Pf_Bonds(idx_fun_x(x_t[1]), idx_fun_a(a_t[1]), idx_fun_state(state_t[1])) #bonds emission when there is no default
            price_t[1]=Pf_Price(idx_fun_x(x_t[1]), idx_fun_a(a_t[2]), idx_fun_state(state_t[1])) #price of the bond emission
        end
    else #default
        def_event_t[1]=1
        a_t[2]=0 #bonds emission in case of default
        price_t[1]=(1+r)^(-1) #price of the bond emission
    end

    #complete iteration
    for t in 2:T
        if Default_t[t-1]==0 ||  Default_t[t-1]==1 && λ_t[t]==1 #did not default in the previous period or defaulted and is now readmitted
            #Decide whether to default or not
            if version=="complete"
                Default_t[t]=[0 1][findmin([abs(0-Pf_Default(idx_fun_x(x_t[t]), idx_fun_a(a_t[t]), idx_fun_z(z_t[t]), idx_fun_g(g_t[t]))) abs(1-Pf_Default(idx_fun_x(x_t[t]), idx_fun_a(a_t[t]), idx_fun_z(z_t[t]), idx_fun_g(g_t[t])))])[2]]
            else
                Default_t[t]=[0 1][findmin([abs(0-Pf_Default(idx_fun_x(x_t[t]), idx_fun_a(a_t[t]), idx_fun_state(state_t[t]))) abs(1-Pf_Default(idx_fun_x(x_t[t]), idx_fun_a(a_t[t]), idx_fun_state(state_t[t])))])[2]]
            end
        
            if Default_t[t]==0 #No default
                if version=="complete"
                    a_t[t+1]=Pf_Bonds(idx_fun_x(x_t[t]), idx_fun_a(a_t[t]), idx_fun_z(z_t[t]), idx_fun_g(g_t[t])) #bonds emission when there is no default
                    price_t[t]=Pf_Price(idx_fun_x(x_t[t]), idx_fun_a(a_t[t+1]), idx_fun_z(z_t[t]), idx_fun_g(g_t[t])) #price of the bond emission          
                else
                    a_t[t+1]=Pf_Bonds(idx_fun_x(x_t[t]), idx_fun_a(a_t[t]), idx_fun_state(state_t[t])) #bonds emission when there is no default
                    price_t[t]=Pf_Price(idx_fun_x(x_t[t]), idx_fun_a(a_t[t+1]), idx_fun_state(state_t[t])) #price of the bond emission
                end
               
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
if report=="mean"
    statistics=[mean(std_y), mean(std_c), mean(std_TBy), mean(std_R)*4, mean(corr_c_y), mean(corr_TBy_Y),
    mean(corr_R_Y), mean(corr_R_TBy), mean(Default_number)*10000, mean(max_R)*4]
elseif report=="median"
    statistics=[median(std_y), median(std_c), median(std_TBy), median(std_R)*4, median(corr_c_y), median(corr_TBy_Y),
    median(corr_R_Y), median(corr_R_TBy), median(Default_number)*10000, median(max_R)*4]
end
