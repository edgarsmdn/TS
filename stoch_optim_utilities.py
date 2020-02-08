import numpy as np
import random as rnd
'''

               Utilities for Stochastic Optimization Algorithms

'''
def LS_f_v3(f, p_init, max_iter, bounds, radius, reduce_iter, reduce_frac):
    '''
    ---------------------
    LOCAL SEARCH ALGORITHM
    ---------------------
    --- input ---
    f: (function) Objective function
    p_init: (array) Initial point (where the funtion is going to be evaluated)
    max_iter: (integer) Maximum number of iterations
    bounds: (list) Bounds on the search domain
    radius: (float) Initial search radius
    reduce_iter: (integer) Number of iterations with the same optimum that will induce a search radius reduction
    reduce_frac: (float) Fraction to which the search radius is reduced. Must be between 0 and 1

    --- output ---
    best_value: (float) The best value found with the iteration using the best_position
    best_position: (array) The best position in which the final best_value was evaluated

    --- Notes ---
    1. f stands for "fast implementation" which means it does not compile results
    '''
    #Initialization
    f = f
    p_init = p_init
    max_iter = max_iter
    bounds = bounds
    radius = radius
    reduce_iter = reduce_iter
    reduce_frac = reduce_frac
    # ----------------------------------------------
    best_position = p_init
    best_value = f(p_init)
    dim = len(p_init)
    fail_count = 0   
    # Iteration Loop
    for i_iter in range(max_iter):
        # Tries random values in the vecinity of the best position so far
        # Assure that every variable is within the bounds
        check = False
        while not check:
            temp_bound = np.array([rnd.uniform(bounds[i][0],bounds[i][1]) for i in range(dim)])
            p_trial = - best_position + radius * temp_bound
            check = check_bounds(bounds, p_trial)
            if not check:
                p_trial = best_position - radius * temp_bound
                check = check_bounds(bounds, p_trial)
            # If the modification of the complete set did not work. It will modify each variable individually
            if not check:
                p_trial = check_bounds_variable(bounds, p_trial, radius)
                check = True
        # If trial value is better than best value, this gets substituted
        trial_value = f(p_trial)
        if trial_value < best_value:
            best_position = p_trial
            best_value  = trial_value
        else:
            fail_count += 1
        # Check whether it's time to set radius to smaller one. Resets failcount
        if fail_count == reduce_iter:
            radius *= reduce_frac
            fail_count = 0
    return best_value, best_position

def LS_p_v3(f, p_init, max_iter, bounds, radius, reduce_iter, reduce_frac):
    '''
    ---------------------
    LOCAL SEARCH ALGORITHM
    ---------------------
    --- input ---
    f: (function) Objective function
    p_init: (array) Initial point (where the funtion is going to be evaluated)
    max_iter: (integer) Maximum number of iterations
    bounds: (list) Bounds on the search domain
    radius: (float) Initial search radius
    reduce_iter: (integer) Number of iterations with the same optimum that will induce a search radius reduction
    reduce_frac: (float) Fraction to which the search radius is reduced. Must be between 0 and 1

    --- output ---
    best_value: (float) The best value found with the iteration using the best_position
    best_position: (array) The best position in which the final best_value was evaluated
    trajectory: (matrix) Column 0: Number of iteration. Column 1: Value for current iteration
    trajectory_x: (matrix) Positions visited during the iterations

    --- Notes ---
    1. The "p" states for the "plot" version of the algorithm. It outputs all the iteration trajectory
    '''
    #Initialization
    f = f
    p_init = p_init
    max_iter = max_iter
    bounds = bounds
    radius = radius
    reduce_iter = reduce_iter
    reduce_frac = reduce_frac
    # ----------------------------------------------
    best_position = p_init
    best_value = f(p_init)
    dim = len(p_init)
    fail_count = 0
    
    trajectory       = np.zeros((max_iter + 1, 2))
    trajectory[0][0] = 0
    trajectory[0][1] = best_value
    
    trajectory_x    = np.zeros((max_iter + 1, dim))
    trajectory_x[0] = best_position
    
    # Iteration Loop
    for i_iter in range(max_iter):
        # Tries random values in the vecinity of the best position so far
        # Assure that every variable is within the bounds
        check = False
        while not check:
            temp_bound = np.array([rnd.uniform(bounds[i][0],bounds[i][1]) for i in range(dim)])
            p_trial = - best_position + radius * temp_bound
            check = check_bounds(bounds, p_trial)
            if not check:
                p_trial = best_position - radius * temp_bound
                check = check_bounds(bounds, p_trial)
            # If the modification of the complete set did not work. It will modify each variable individually
            if not check:
                p_trial = check_bounds_variable(bounds, p_trial, radius)
                check = True
        # If trial value is better than best value, this gets substituted
        trial_value = f(p_trial)
        if trial_value < best_value:
            best_position = p_trial
            best_value  = trial_value
        else:
            fail_count += 1
        # Check whether it's time to set radius to smaller one. Resets failcount
        if fail_count == reduce_iter:
            radius *= reduce_frac
            fail_count = 0
        # Stores trajectory
        trajectory[i_iter + 1][0] = i_iter + 1
        trajectory[i_iter + 1][1] = best_value
        trajectory_x[i_iter + 1]  = best_position
        
        
    return best_value, best_position, trajectory, trajectory_x

def RS_f_v2(f, p_best, max_iter, bounds):
    '''
    ---------------------
    RANDOM SEARCH ALGORITHM
    ---------------------
    --- input ---
    f: objective function
    p_best: hot-start a best point
    max_iter: maximum number of iterations
    bounds: bounds on the search domain
    
    --- output ---
    best_value: (float) The best value found with the iteration using the best_position
    best_position: (array) The best position in which the final best_value was evaluated

    --- Notes ---
    1. p_best is used in case a good value is already known
    2. The "f" states for the "fast" version of the algorithm. It only outputs the best values found
    '''
    # Initialization
    f = f
    p_best = p_best
    max_iter = max_iter
    bounds = bounds
    # ----------------------------------------------
    best_position = p_best
    best_value = f(p_best)
    dim = len(p_best)
    # Search loop
    for i_iter in range(max_iter):
        # Tries random values
        p_trial = np.array([rnd.uniform(bounds[i][0],bounds[i][1]) for i in range(dim)])
        trial_value = f(p_trial)
        # If trial values is better than best position, this gets substituted
        if trial_value < best_value:
            best_position = p_trial
            best_value  = trial_value
    return best_value, best_position

def RS_p_v2(f, p_best, max_iter, bounds):
    '''
    ---------------------
    RANDOM SEARCH ALGORTHM
    ---------------------
    --- input ---
    f: objective function
    p_best: hot-start a best point
    max_iter: maximum number of iterations
    bounds: bounds on the search domain

     --- output ---
    best_value: (float) The best value found with the iteration using the best_position
    best_position: (array) The best position in which the final best_value was evaluated
    trajectory: (matrix) Column 0: Number of iteration. Column 1: Value for current iteration
    
    --- Notes ---

    1. p_best is used in case a good value is already known.
    2. The "p" states for the "plot" version of the algorithm. It outputs all the iteration trajectory

    '''
    # Initialization
    f = f
    p_best = p_best
    max_iter = max_iter
    bounds = bounds
    # ----------------------------------------------
    best_position = p_best
    best_value = f(p_best)
    dim = len(p_best)
    # Creating arrays for the plots
    all_results = np.zeros((max_iter,2))
    # Search loop
    for i_iter in range(max_iter):
        # Tries random values 
        p_trial = np.array([rnd.uniform(bounds[i][0],bounds[i][1]) for i in range(dim)])
        trial_value = f(p_trial)
        # If trial values is better than best position, this gets substituted
        if trial_value < best_value:
            best_position = np.copy(p_trial)
            best_value  = trial_value
        # Compiling results
        all_results[i_iter][0] = i_iter
        all_results[i_iter][1] = best_value
    return best_value, best_position, all_results

def check_bounds_variable(bounds, position, radius):
    '''
    ------------------------------
    CHECK BOUNDS VARIABLE BY VARIABLE AND ASSURES THEY ARE WITHIN BOUNDS CHANGING THEM WHEN NECESSARY
    ------------------------------
    --- input ---
    bounds: (list) Bounds on the search domain
    position: (array) Proposed current position of the particle
    
    --- output ---
    position: (array) Corrected array to be within bounds in each variable
    '''
    # Initialization
    bounds = bounds
    position = position
    radius = radius
    # ----------------------------------------------
    check = False
    while not check:
        check_var_count = 0 #To count variables which are within bounds
        for variable in range(len(position)):
            bounds_variable = [bounds[variable]] # Extracts the bounds for the specific variable
            check_variable = check_bounds(bounds_variable, np.array([position[variable]]))
            if not check_variable:
                r1 = variable - radius # Left limit radius 
                r2 = variable + radius # Right limit radius 
                
                if r2 < bounds_variable[0][0]:                                  # O /------/
                    position[variable] = bounds_variable[0][0]
                elif r1 > bounds_variable[0][1]:                                # /------/ O
                    position[variable] = bounds_variable[0][1]
                elif r2 > bounds_variable[0][0] and r1 < bounds_variable[0][0]: # O----/
                    position[variable] = rnd.uniform(bounds_variable[0][0], r2)
                elif r1 < bounds_variable[0][1] and r2 > bounds_variable[0][1]: # /----O
                    position[variable] = rnd.uniform(r1, bounds_variable[0][1])
                elif r1 > bounds_variable[0][0] and r2 < bounds_variable[0][1]: # /--O--/
                    position[variable] = rnd.uniform(r1, r2)
                
                check_variable = check_bounds(bounds_variable, np.array([position[variable]]))
            if check_variable:
                check_var_count += 1
        if check_var_count == len(position):
            check = True
    if check:
        return position

def check_bounds(bounds, position):
    '''
    ------------------------------
    CHECK BOUNDS ALGORITM
    ------------------------------
    --- input ---
    bounds: (list) Bounds on the search domain
    position: (array) Proposed current position of the particle
    
    --- output ---
    valid_position: (boolean) "True" if position is within the allowed bounds in every dimension and "False" otherwise
    '''
    # Initialization
    bounds = bounds
    position = position
    # ----------------------------------------------
    dim = len(bounds)
    count = 0
    for i in range(dim):
        if position[i] <= bounds[i][1] and position[i] >= bounds[i][0]:
            count += 1
    if count == dim:
        return True
    else:
        return False

def tabu_zone(tabu, continuos_radius):
    '''
    ------------------------------
    DEFINES TABU ZONE FOR EACH VARIABLE IN A POINT GIVEN A CONTINUOS RADIUS
    ------------------------------
    --- input ---
    tabu: (array) Point classified as Tabu
    continuos_radius: (list) Radius around each variable to define tabu zone for each variable in the tabu point
   
    --- output ---
    tabu_z: (list) It contains the tabu zone per each variable of the point
    '''
    # Initialization
    tabu = tabu
    continuos_radius
    # ----------------------------------------------
    tabu_z = [] # To store tabu zones for each variable
    # Defines tabu zone per each variable
    for i in range(len(tabu)):
        left = tabu[i] - continuos_radius[i] # Defines left bound of the tabu zone
        right = tabu[i] + continuos_radius[i] # Defines right bound of the tabu zone
        tabu_z.append((left, right))
    return tabu_z

def tabu_zones(tabuList, continuos_radius):
    '''
    ------------------------------
    GIVES TABU ZONES FOR EACH TABU IN THE LIST
    ------------------------------
    --- input ---
    tabuList: (list) Stores all the current tabu points
    continuos_radius: (list) Radius around each variable to define tabu zone for each variable in the tabu point
   
    --- output ---
    tabu_zs: (list) It contains the tabu zones per each tabu point
    '''
    # Initialization 
    tabuList = tabuList
    continuos_radius = continuos_radius
    # ----------------------------------------------
    tabu_zs = []
    for tabu in tabuList:
        t_z = tabu_zone(tabu, continuos_radius)
        tabu_zs.append(t_z)
    return tabu_zs

def check_tabu(position, tabu_zs):
    '''
    ------------------------------
    CHECKS IF A POINT IS A TABU OR NOT
    ------------------------------
    --- input ---
    position: (array) Position within the search space
    tabu_zs: (list) It contains the tabu zones per each tabu point
   
    --- output ---
    tabu: (boolean) "True" if position is a tabu (it's within the tabu zone in each one of its variables) and "False" otherwise
    '''
    # Initialization
    position = position
    tabu_zs = tabu_zs
    # ----------------------------------------------
    for tabu_z in tabu_zs:
        check = check_bounds(tabu_z, position) # Checks if the position is within a tabu zone
        if check:
            return True
    if not check:
        return False

def p_outside_tabu(bounds, tabu_zs):
    '''
    ------------------------------
    GENENRATES A POSITION OUTSIDE TABU ZONES
    ------------------------------
    --- input ---
    bounds: (list) Bounds on the search domain
    tabu_zs: (list) It contains the tabu zones per each tabu point
   
    --- output ---
    p_out_tabu: (array) Random particle outside tabu zones and within bounds
    '''
    # Initialization
    bounds = bounds
    tabu_zs = tabu_zs
    # ----------------------------------------------
    p_out_tabu = np.zeros(len(bounds)) # Creates an array of zeros to store new position
    
    check = True
    while check:
        select_tabu_z = rnd.choice(tabu_zs) # Selects randomnly one of the sets of tabu zones
        for i in range(len(bounds)):
            # Selects valid limits
            if select_tabu_z[i][0] > bounds[i][0] and select_tabu_z[i][1] < bounds[i][1]:    # /--O--/
                left1 = bounds[i][0]
                right1 = select_tabu_z[i][0]
                left2 = select_tabu_z[i][1]
                right2 = bounds[i][1]
            elif select_tabu_z[i][0] > bounds[i][0] and select_tabu_z[i][1] >= bounds[i][1]:  # /----O
                left1 = bounds[i][0]
                right1 = select_tabu_z[i][0]
                left2 = left1
                right2 = right1
            elif select_tabu_z[i][0] <= bounds[i][0] and select_tabu_z[i][1] < bounds[i][1]:  # O----/
                left1 = select_tabu_z[i][1]
                right1 = bounds[i][1]
                left2 = left1
                right2 = right1
            new_bounds = rnd.choice(((left1, right1),(left2, right2)))
            p_out_tabu[i] = rnd.uniform(new_bounds[0], new_bounds[1])
        check = check_tabu(p_out_tabu, tabu_zs) # Checks whether the new position is a tabu or not
    return p_out_tabu
             
def new_particle(bounds):
    '''
    ------------------------------
    GENERATES NEW PARTICLE (POINT) RANDOMLY
    ------------------------------
    --- input ---
    bounds: (list) Bounds on the search domain
   
    --- output ---
    particle: (array) Random particle within bounds
    '''
    #Initialization
    B = bounds
    # ----------------------------------------------
    dim = len(B)
    #Generate new random particle within the bounds
    particle = np.array([rnd.uniform(B[i][0],B[i][1]) for i in range(dim)])
    return particle

def first_generation(num_p, bounds):
    '''
    ------------------------------
    GENERATES FIRST GENERATION FOR GA
    ------------------------------
    --- input ---
    num_p: (integer) Number of particles in the new generation to be created
    bounds: (list) Bounds on the search domain
    
    --- output ---
    generation: (list) Set of new particles
    '''
    # Initialization
    S = num_p
    B = bounds
    # ----------------------------------------------
    generation = []
    # Generates a set of num_p new particles
    for point in range(S):
        particle = new_particle(B)
        generation.append(particle)
    return generation

def sort_standard(f, generation):
    '''
    ------------------------------
    STANDARD SORT (WALKS THROUGH EACH ELEMENT IN LIST)
    ------------------------------
    --- input ---
    f: (function) Objetive function
    generation: (list) Set of new particles
    
    --- output ---
    g_sorted: (matrix) Sorted set of new particles. Row: particle, Column: variable
    '''
    # Initialization
    F = f
    G = generation
    # ----------------------------------------------
    dim = len(G)
    num_var = len(G[0])
    g_sorted = np.reshape([(0.0) for i in range(num_var*dim)], (dim,num_var)) # Creates a matrix of zeros
    values = np.zeros((dim,2)) # Creates a matrix of zeros
    # Stores the points with their respective value(as a key)
    index = 0
    for particle in G:
        values[index][0] = F(particle)
        values[index][1] = index
        index += 1
    # Sorts values
    values_sorted = values[np.argsort(values[:,0])]
    # Stores sorted values in the previously created matrix
    ind_sorted = values_sorted[:,1]
    i = 0
    for ind in ind_sorted:
        g_sorted[i] = G[int(ind)]
        i += 1
    return g_sorted

def selection(g_sorted, best_num, random_num):
    '''
    ------------------------------
    SELECTION OF THE FITTEST POINTS AND SOME RANDOME ONES
    ------------------------------
    --- input ---
    g_sorted: (matrix) Sorted set of new particles. Row: particle, Column: variable.
    best_num: (integer) Number of best particles you want to select
    random_num: (integer) Number of random particles you want to select from the rest
    
    --- output ---
    selected: (matrix) Set of particles selected
    '''
    #Initialization
    g = g_sorted
    best = best_num
    random = random_num
    # ----------------------------------------------
    num_var = len(g[0])
    selected = np.reshape([(0.0) for i in range(num_var*(best + random))], ((best + random),num_var)) # Creates a matrix of zeros
    # Stores the best points to the matrix "selected"
    for i in range(best):
        selected[i] = g[i]
    # Stores points from the rest of the generation randomly  
    for i in range(random):
        selected[i + best] = rnd.choice(g[best:])
    return selected

def define_parents(selected, parents_child):
    '''
    ------------------------------
    SELECTION OF POINTS WHICH ARE GONNA BE RECOMBINED AMONG THEM
    ------------------------------
    --- input ---
    selected: (matrix) Set of particles selected
    parents_child: (integer) Number of parents per child
   
    --- output ---
    groups_par: (list) Set of groups of parents that are gonna be recombined
    '''
    # Initialization
    parents = selected
    N = parents_child
    # ----------------------------------------------
    groups_par = []
    # Loop to define parents
    row_parent = 0 # Number of the row in the Matrix for the current parent in the next loop
    for parent in parents:
        group_repro = np.zeros((N,len(parent))) # Creates a matrix of zeros
        # Makes a matrix of the candidates to be reproduced with
        candidates = np.delete(parents, row_parent, 0)
        # Randomly select the parents from candidates to be later reproduce with.
        for i in range (N-1):
            cand = rnd.choice(candidates)
            group_repro[i] = parent
            group_repro[i+1] = cand
            index_row = np.where([cand] == [cand])[0][0] # Gives de row number of "cand"
            candidates = np.delete(candidates, index_row, 0) # Prevent repetition within the group of parents
        groups_par.append(group_repro)
        row_parent += 1
    return groups_par
    
def reproduction(groups_par, num_children):
    '''
    ------------------------------
    REPRODUCTION OF PARENTS BY RANDOM RECOMBINATION
    ------------------------------
    --- input ---
    groups_par: (list) Set of groups of parents that are gonna be recombined
    num_children: (integer) Number of children per group of parents
    
    --- output ---
    new_gener_r: (matrix) Set of children (points) produced by recombination of the groups of parents
    '''
    # Initialization
    groups = groups_par
    num_ch = num_children
    # ----------------------------------------------
    new_gener = []
    num_ac_child = 0 #Number of current children per group of parents 
    # Produces the number of children specified per group of parents
    while num_ac_child < num_ch:
        # Loops per group of parents
        for group in groups:
            child = []
            #Loops per each variable in a point 
            for variable in range(len(group[0])):
                can_var = []
                # Selects the variable randomly between the group of parents
                for i in range(len(group)):
                    can_var.append(group[i][variable])
                child.append(rnd.choice(can_var))
            new_gener.append(child)
        num_ac_child += 1
    new_gener_r = np.asarray(new_gener)
    return new_gener_r

def mutation(new_gener_r, bounds, continuos_radius):
    '''
    ------------------------------
    MUTATION OF NEW CHILDREN WITH CERTAIN PROBABILITY
    ------------------------------
    --- input ---
    new_gener_r: (matrix) Set of children (points) produced by recombination of the groups of parents
    bounds: (list) Bounds on the search domain
    continuos_radius: (list) Radius around each variable to define prohibeted zone for each variable. Continuos numbers application
    
    --- output ---
    g_m: (matrix) Set of children (points) passed through random mutation with certain probability
    '''
    #Initialization
    g_r = new_gener_r
    B = bounds
    c_r = continuos_radius
    # ----------------------------------------------
    num_var = len(g_r[0])
    # Makes a random change in a random variable per child with certain probability of mutation
    i = 0
    for child in g_r:
        probability = 1/num_var # 1/num of decision variables according to Deb,K. (2001), Multi-Objective Optimization Using Evolutionary Algorithms
        if child in np.delete(g_r, i): # Prevents a child to be exactly the same as one of its parents
            probability = 1
        if rnd.random() < probability:
            random_index = np.where( child == (rnd.choice(child)))[0][0]
            c_r_v = [c_r[random_index]] # Selects the continuos radius to the chosen variable to be mutated and store it as need it for "tabu_zones"
            var_mut = [np.array([child[random_index]])] # Selects the chosen variable to be mutated and store it as need it for "tabu_zones"
            proh_zone = tabu_zones(var_mut, c_r_v) # Defines the zone within the continuos radius
            bounds_var = [B[random_index]] # Selects the bounds for the chosen variable to be mutated
            mutated_variable = p_outside_tabu(bounds_var, proh_zone) # Gives a random number for the chosen variable outside continuos radius and within bounds
            child[random_index] = mutated_variable # Replaces the mutated variable in place
            #child[random_index] = rnd.uniform(B[random_index][0],B[random_index][1])
        i += 1
    new_gener_m = g_r
    return new_gener_m
