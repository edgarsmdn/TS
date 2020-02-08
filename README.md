# Tabu Search

<p align="center">
<img src="https://github.com/edgarsmdn/TS/blob/master/TS_1.gif" width="500"> 
</p>

## Development

This work was part of the project I did during my undergrad research internship in the summer of 2018 at the [Centre for Process Integration](https://www.ceas.manchester.ac.uk/cpi/), The University of Manchester on stochastic optimization.

## Background

Tabu Search (TS) is an extension of the Local Search (LS) optimization algorithm and it is accredited to Fred Glover (Glover, 1986). Likewise, TS starts with a random point in the search space and try to reach the local minimum in the vicinity of the point. Once the LS stage finishes, it stores the best point found in a list (known as “Tabu List”) and repeat the search process with a new random point. As the name suggests, the tabu list prevents the algorithm to choose the same point twice.

TS is usually used for discrete points but it can be extended for continuous functions if a tabu continuous radius is implemented. In this case, the Tabu List will prevent the algorithm to select points within the vecinity defined by the continuous radius. In other words, instead of preventing single points to be selected, the augmented algorithm prevents the sampling in the neighborhoods of the already visited points. The Tabu list is updated once its final length (which the user has to determine) is reached, and it deletes the oldest tabu to be able to store the new one. In this way, the memory requirement is small and the algorithm can walk towards the optimum.

## Prerequisites

The function requires Python 3.0 (or more recent versions).

## Functioning

#### Inputs

```
TS(f, max_iter, max_iter_LS, bounds, radius, reduce_iter, reduce_frac, max_tabu_size, continuos_radius, p_init=None, traj=False)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. The **function** to be optimized. The functions needs to be of the form ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5En%20%5Crightarrow%20%5Cmathbb%7BR%7D).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. The **maximum number of iterations** in the Tabu Search (stopping criteria).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. The **maximum number of iterations for the sub-routine Local Search**. Local Search is used at each non-Tabu proposed point to find the local optimum close to it.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. The **bounds** for each dimension of the fucntion. This has to be a list of the form `[(lb1, ub1), (1b2, ub2), ...]`.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5. The **initial search radius** for Local Search.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6. The **number of iterations with the same optimum** that will induce a search radius reduction in LS.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 7. The **fraction to which the search radius is reduced in LS**.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 8. The **maximum number of points stored**  in the Tabu List.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 9. The **Tabu radius** which defines the Tabu neighborhoods.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 9. (Optional) The **initial point** for the funtion to be firstly evaluated. If not given, it will generates one randomly

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 10. (Optional) The **ON/OFF** feature for the algorithm to provide the trajectory. Default is false.

#### Outputs

```
Optimum: (class) Results with:
        Optimum.f: (float) The best value of the funtion found in the optimization
        Optimum.x: (array) The best point in which the function was evaluated
        Optimum.traj_f: (array) Trajectory of function values (including LS iterations)
        Optimum.traj_x: (array) Trajectory of positions (including LS iterations)
        Optimum.traj_f_sum_up: (array) Trajectory of function values (including TS steps only). 
                                        First column is number of total iterations.
        Optimum.traj_x_sum_up: (array) Trajectory of positions (including TS steps only)
```

#### General information

* The current stopping criteria is the maximum number of iterations specified by the user.

* Within the Local Search (LS) framework that is performed at each major iteration of TS, the search neighborhood is reduced iteratively by ![eq](https://latex.codecogs.com/gif.latex?r%5E%7Bt&plus;1%7D%20%3D%20r_%7Bred%7D%20%7E%20r%5Et). A LS is performed at each major iteration of the Tabu Search algorithm.

* The Local Search algorithm is contained in the utilities file *stoch_optim_utilities.py* 

### References

Glover, F. (1986). Future Paths for Integer Programming and Links to Artificial Intelligence. Computers and Operations Research Vol. 13, 533-549.

## License

This repository contains a [MIT LICENSE](https://github.com/edgarsmdn/TS/blob/master/LICENSE)
