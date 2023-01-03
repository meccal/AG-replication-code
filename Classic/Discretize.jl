"""
LUCA MECCA
lmecca@london.edu
September 2022
"""

#Function Tau discretizes AR(1) processes into discrete grids using the approach proposed in Tauchen (1986)
#It also computes the corresponding transition matrix
#Inputs:
#ρ: autoregressive coefficient.
#σ: volatility of the shock.
#K: number of grid points. Default value is 9.
#λ: MC approx parameter. Default value is 3 (as suggested in Tauchen).
#interc: fix part. Default value is 0.

function Tau(ρ::Float64, σ::Float64, K::Int64=9, λ::Number=3, interc::Number=0)
    #Initial point
    z_1=-λ*σ/sqrt(1-ρ^2)
    #size of the step
    step=-2*z_1/(K-1)
    #create the grid
    grid=Array{Number}(undef,K,1) 

    grid[1]=z_1  
    #complete the grid
    for i in 2:K
        grid[i]=z_1+(i-1)*step
    end 

    #add the constant part
    grid=grid .+ interc

    #Now create the transition matrix
    #T_ij indicates the probability of moving from state j to state i
    #For example the first column of the matrix includes the probability of moving from
    #the first state to itself and all the other states
    #The first row includes the probability of going from all the states to the first state.
    T_matrix = Matrix{Float64}(undef,K,K) 
    #loop over each column of the transition matrix
    for col in 1:K
        #probability of moving from any of the states to state 1 (i.e first row of the transition matrix)
        T_matrix[1,col]=cdf(Normal(),(grid[1]-interc-grid[col]*ρ)/σ+step/2/σ)
        #Compute the transition probability of in-betweeen states 
        for row in 2:(K-1)
            T_matrix[row,col]=cdf(Normal(),(grid[row]-interc-grid[col]*ρ)/σ+step/2/σ)-cdf(Normal(),(grid[row]-interc-grid[col]*ρ)/σ-step/2/σ)
        #probability of moving from any of the states to state m (i.e last row of the transition matrix)
        T_matrix[K,col]=max(1-sum(T_matrix[1:K-1,col]),0)
        end
    end
    return grid, T_matrix
end

#This function is used to discretize log consumption growth (which is not an AR(1) process and, therefore, we cannot use the MC function above)
#It creates the grid as equally spaced values in a range that includes 3 standard deviation shocks from each side
#It also computes the probability that each shock happens
#Inputs:
#x_t: time-varying part
#σ: volatility of the shock.
#β: coefficient on the time-varying part. Default value is 1.
#K: number of grid points. Default value is 9.
#interc: fix part. Default value is 0.

function discr(x_t::Number,  σ::Float64, K::Int64=9, interc::Number=0, β::Number=1)
    lower_bound=interc+β*x_t-3*σ #lower bound
    upper_bound=interc+β*x_t+3*σ #upper bound
    grid=LinRange(lower_bound, upper_bound, K)

    #Now compute the probability associated to each shock
    prob_shock=Array{Float64}(undef, K, 1)
    step=(grid[K]-grid[1])/(K-1) #length of the step
    #initialize
    prob_shock[1]=cdf(Normal(),(grid[1]-interc-β*x_t)/σ+step/2/σ)
    #complete the grid
    for i in 2:(K-1)
        prob_shock[i]=cdf(Normal(),(grid[i]-interc-β*x_t)/σ+step/2/σ)-cdf(Normal(),(grid[i]-interc-β*x_t)/σ-step/2/σ)
    end
    prob_shock[K]=max(1-sum(prob_shock[1:K-1]),0)

    return grid, prob_shock

end

