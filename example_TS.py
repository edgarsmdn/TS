import numpy as np
from TS import TS
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

'''

                              Example of TS 

'''

def alpine1(variables):
    '''
    Alpine 1 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_A.html#n-d-test-functions-a
    Retrieved: 19/06/2018
    '''
    return np.sum(np.abs(np.multiply(variables, np.sin(variables)) + 0.1 * variables))

f = alpine1
b = (-10, 10)

max_iter = 10
max_iter_LS = 5
bounds=[b for i in range(2)]
radius = 0.5
reduce_iter = 100
reduce_frac = 0.8
max_tabu_size = 20
continuos_radius = [(0.01) for i in range(len(bounds))]


results = TS(f, max_iter, max_iter_LS, bounds, radius, reduce_iter, 
             reduce_frac, max_tabu_size, continuos_radius, p_init=None, traj=True)


#print('Best function value: ', results.f)
#print(' ')
#print('Best point: ', results.x)
#print(' ')
#print('Trajectory function values: ', results.traj_f)
#print(' ')
#print('Trajectory points: ', results.traj_x)
#print(' ')
#print('Trajectory function values summed up: ', results.traj_f_sum_up)
#print(' ')
#print('Trajectory points summed up: ', results.traj_x_sum_up)


# Plot Optimization
points             = np.zeros(1, dtype=[("position", float, 2)])
points["position"] = results.traj_x[0]

fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.15, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 
start, stop, n_values = b[0], b[1], 100

x       = np.linspace(start, stop, n_values)
y       = np.linspace(start, stop, n_values)
X, Y    = np.meshgrid(x, y)

zs = np.array([f(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z  = zs.reshape(X.shape)

cm = plt.contourf(X, Y, Z, cmap='Blues')
plt.colorbar(cm)
ax.set_title('Alpine 1 function')
ax.set_xlabel('x')
ax.set_ylabel('y')

scatter = ax.scatter(points["position"][:,0], points["position"][:,1], c='red', s=50)

xs = results.traj_x

def update(frame_number):
    points["position"] = xs[frame_number]
    ax.set_title('Alpine 1 function, Iteration: ' + str(frame_number))
    scatter.set_offsets(points["position"])
    return scatter, 

anim = FuncAnimation(fig, update, interval=0.1, frames=range(len(xs)), repeat_delay=2000)
plt.show()

# Save gif
anim.save('TS.gif', writer='imagemagick', fps=400)