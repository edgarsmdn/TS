import numpy as np
import random as rnd
from stoch_optim_utilities import LS_p_v3
from stoch_optim_utilities import p_outside_tabu
from stoch_optim_utilities import check_bounds_variable
from stoch_optim_utilities import check_tabu
from stoch_optim_utilities import tabu_zones

def TS(f, max_iter, max_iter_LS, bounds, radius, reduce_iter, reduce_frac, max_tabu_size, continuos_radius, p_init=None, traj=False):
    '''
    ------------------------------
    TABU SEARCH ALGORITHM
    ------------------------------
    --- input ---
    f: (function) Objetive function
    max_iter: (integer) maximum number of iterations for Tabu Search
    max_iter_LS: (integer) maximum number of iterations for subroutine "Local Search (LS)"
    bounds: (list) bounds on the search domain
    radius: (float) initial search radius for Local Search
    reduce_iter: (integer) number of iterations with the same optimum that will induce a search radius reduction in LS
    reduce_frac: (float) fraction to which the search radius is reduced in LS
    max_tabu_size: (integer) Maximum number of points stock in the memory for Tabu Search
    continuos_radius: (list) This radius will categorize as Tabu a point within the zone between this radius and the current position
    p_init: (array) Initial point for the funtion to be firstly evaluated. If not given, it will generates one randomly
    traj: (boolean) To output trajectory or not. Default is false
    
    --- output ---
    f(sBest): (float) The best value of the funtion found in the optimization
    sBest: (array) The best point in which the function was evaluated
    trajectory: (matrix) Column 0: Number of iteration. Column 1: Value for current iteration. It stores the complete trajectory including LS iterations 
    trajectory_sum_up: (matrix) Column 0: Number of iteration. Column 1: Value for current iteration. It stores only the iterations from TS but with the real number of iteration
    '''
    # Initialize
    dim = len(bounds)
    # Generates a random p_init if not given
    if np.any(p_init) == None:
        p_init = np.array([rnd.uniform(bounds[i][0],bounds[i][1]) for i in range(dim)])
    
    sBest           = p_init
    bestCandidate   = p_init
    iteration       = 0
    tabuList        = []
    tabuList.append(bestCandidate)
     
    if traj:
        # To store summed up of trajectory
        trajectory_sum_up       = np.zeros((max_iter + 1, 2))
        trajectory_sum_up[0][0] = 0
        trajectory_sum_up[0][1] = f(sBest)
        traj_x_sum_up           = np.zeros((max_iter + 1, dim))
        traj_x_sum_up[0]        = sBest
        
        # To store complete trajectory
        trajectory       = np.zeros(max_iter * (max_iter_LS + 1)+ 1) 
        trajectory[0]    = f(sBest)
        traj_x           = np.zeros((max_iter * (max_iter_LS + 1)+ 1, dim))
        traj_x[0]        = sBest
    
        beginning = 1 # Helps to store complete trajectory properly
    
    # Iteration Loop
    while iteration < max_iter:        
        list_dim = len(tabuList)
        # Checks if bestCandidate is within bounds, if not, it will give a point within them
        bestCandidate = check_bounds_variable(bounds, bestCandidate, radius)
        # Check if the Candidate is a Tabu or not
        tabu_zs = tabu_zones(tabuList, continuos_radius) # Defines tabu zones based on the tabu list and continuos radius
        Tabu    = check_tabu(bestCandidate, tabu_zs)
        # Decide what to do if the Candidate is a Tabu
        if Tabu: # If Tabu, it assigns another random neighbour value of the current bestCandidate
            bestCandidate = p_outside_tabu(bounds, tabu_zs) # Generates a point outside the tabu zones and within bounds
            local_search  = LS_p_v3(f, bestCandidate, max_iter_LS, bounds, radius, reduce_iter, reduce_frac) # Searches the local minimum around bestCandidate
            bestCandidate = local_search[1]
            tabuList.append(bestCandidate)
        else: # If not Tabu, it makes the local search around this point to try to reach the local minimum
            local_search  = LS_p_v3(f, bestCandidate, max_iter_LS, bounds, radius, reduce_iter, reduce_frac)
            bestCandidate = local_search[1]
            tabuList.append(bestCandidate)
        # Updates information
        if f(bestCandidate) < f(sBest):
            sBest = bestCandidate
        # Checks whether is time to updates the Tabu list
        if list_dim > max_tabu_size:
            tabuList = tabuList[1:]
        iteration += 1
        
        if traj:
            # Stores complete trajectory
            trajectory[beginning:(iteration*(max_iter_LS + 1) + 1)] = local_search[2][:,1] # Copies the results from LS in trajectory
            traj_x[beginning:(iteration*(max_iter_LS + 1) + 1)]     = local_search[3]
            
            # Replace last value with the result of TS
            trajectory[iteration*(max_iter_LS + 1)] = f(sBest)
            traj_x[iteration*(max_iter_LS + 1)]     = sBest
            
            beginning = ((iteration)*(max_iter_LS + 1) + 1 )
            
            # Stores summed up trajectory
            trajectory_sum_up[iteration][0] = ((iteration)*(max_iter_LS + 1))
            trajectory_sum_up[iteration][1] = f(sBest)
            traj_x_sum_up[iteration]        = sBest
    
    # Gather results
    class Optimum:
        pass    
    
    if traj:
        Optimum.f                = f(sBest)
        Optimum.x                = sBest
        Optimum.traj_f           = trajectory
        Optimum.traj_x           = traj_x
        Optimum.traj_f_sum_up    = trajectory_sum_up
        Optimum.traj_x_sum_up    = traj_x_sum_up
        
    else:
        Optimum.f                = f(sBest)
        Optimum.x                = sBest
        
    return Optimum