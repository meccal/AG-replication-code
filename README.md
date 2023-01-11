# AG-replication-code
This code replicates Aguiar and Gopinath (2006) in julia. We solve the model with the same approach followed by the authors: state space discretization 
and value function iteration on a detrended version of the model. Please see the technical appendix of Aguiar and Gopinath (2006) for details.
The original code from the authors is available here https://scholar.harvard.edu/gopinath/publications/defaultable-debt-interest-rates-and-current-account. 

Folders
Classic: this folder replicates Aguiar and Gopinath (2006) with risk neutral lenders.
Risk Averse: this folder replicare Aguiar and Gopinath (2006) with risk averse lenders with EZ preferences.

Files
AG_VFI_Discrete.jl/AG_VFI_RA.jl: performs the value functions iteration and finds the fixed points of value and policy functions.
Simulations_Discrete.jl/Simulations_RA.jl: simulates a number of paths of the model and computes statistics of interest.
Functions.jl: contains all the performance-sensitive functions to increase the speed of the overall process.
Discretize.jl: contains the functions that are used to discretize the state space (particularly the Tauchen algorithm).

For any question or comments, you can reach me at lmecca@london.edu
