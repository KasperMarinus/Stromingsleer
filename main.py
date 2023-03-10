#%% Imports
import numpy as np, matplotlib.pyplot as plt, time

#%% Main functions
# A function to generate the points on a circle around a given center and with a given radius
def generate_circle(center, radius, steps):
    # We calculate the points using complex powers of e, and return a set of x- and y-coordinates
    coords = np.array([center + radius*np.exp(gamma*1j) for gamma in np.linspace(0, 2*np.pi, steps, endpoint=False)])
    return coords.real, coords.imag


# A function to apply the Joukowski transform to given coordinates. The input could be both complex coordinates or a tuple of real coordinates,
# and the output will match this type
def transform(coords, b=1):
    match coords:  # requires Python 3.10
        case x_coords, y_coords:  # if we got a set of x- and y-coordinates:
            complex_coords = x_coords + 1j*y_coords  # make these into one or multiple complex coordinates
            transformed_complex_coords = complex_coords + b**2/complex_coords  # apply the Joukowski transform
            return transformed_complex_coords.real, transformed_complex_coords.imag  # return the result as x- and y-coordinates
        case complex_coords:  # if we got (one or multiple) complex coordinates:
            return complex_coords + b**2/complex_coords  # return the complex result as complex coordinates


# A function to calculate the complex potential at given coordinates
def complex_potential(coords, gamma, u=1, offset=-.1+.22j):
    match coords:  # requires Python 3.10
        case x_coords, y_coords:  # if the input was real:
            complex_coords = x_coords + y_coords*1j  # transform into complex coordinates
            return u*(complex_coords-offset) + u/(complex_coords-offset) -1j*gamma/(2*np.pi)*np.log((complex_coords-offset))  # return the complex potential
        case complex_coords:  # if the input was complex:
            return u*(complex_coords-offset) + u/(complex_coords-offset) -1j*gamma/(2*np.pi)*np.log((complex_coords-offset))  # return the complex potential


# A function to calculate the streamfunction at given coordinates
def streamfunction(coords, gamma):
    return complex_potential(coords, gamma).imag  # return the imaginary part of the complex potential

#%% Exercise a
Z_0, R = -.1 + .22j, 1.12  # define the center of the circle and its radius (as given in the exercise)

coords = generate_circle(Z_0, R, 1000)  # generate the circle with the given conditions
x_coords_transformed, y_coords_transformed = transform(coords)  # apply the Joukowski transform

# Plot the results
plt.plot(x_coords_transformed, y_coords_transformed)
plt.axis('equal')
plt.show()

#%% Exercise b
# A function to display the streamfunction as a contourplot on a given grid for a certain gamma
def display_streamfunction(gridpoints, gamma, levels=15):
    values = streamfunction(gridpoints, gamma)  # calculate the streamfunction along the grid
    
    # Plot the results
    plt.contourf(gridpoints[0], gridpoints[1], values, levels)
    plt.axis('equal')
    plt.colorbar()
    plt.title(fr"The streamfunction for $\gamma=${gamma}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# Define the linear spaces for r and gamma and make it into a meshgrid
r, gam = np.meshgrid(np.linspace(1.12, 5, 500), np.linspace(0, 2*np.pi, 500))

X, Y = r*np.cos(gam) - .1, r*np.sin(gam) + .22  # transform the meshdrid into the X-Y plane

display_streamfunction((X, Y), 0)  # display the streamfunction for gamma = 0
display_streamfunction((X, Y), -3)  # display the streamfunction for gamma = -3

#%% Exercise c
def display_transformed_streamfunction(gridpoints, gamma, levels=15):
    values = streamfunction(gridpoints, gamma)  # calculate the streamfunction along the grid
    
    X_transform, Y_transform = transform(gridpoints)  # apply the Joukowski transform to the grid

    # Plot the results
    plt.contourf(X_transform, Y_transform, values, levels)
    plt.axis('equal')
    plt.colorbar()
    plt.title(fr"The streamfunction for $\gamma=${gamma}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

display_transformed_streamfunction((X, Y), 0)  # display the streamfunction for gamma = 0
display_transformed_streamfunction((X, Y), -3)  # display the streamfunction for gamma = -3

#%% Exercise d
r2, gam2 = np.meshgrid(np.linspace(1.12, 1.2, 100), np.linspace(-.5, .1, 100))

X2, Y2 = r2*np.cos(gam2) - .1, r2*np.sin(gam2) + .22

display_transformed_streamfunction((X2, Y2), -3)

#%% Exercise e
r_zoomed, gam_zoomed = np.meshgrid(np.linspace(1.12, 2, 100), np.linspace(-1,1,100))
X_zoomed, Y_zoomed = r_zoomed*np.cos(gam_zoomed) - .1, r_zoomed*np.sin(gam_zoomed) + .22

for i in np.linspace(-2.5,-2.4,20):
    display_transformed_streamfunction((X_zoomed, Y_zoomed), i)

#%% Exercise f
optimal_circulation = -2.435
optimal_potential = complex_potential((X, Y), optimal_circulation)

def compute_velocity(grid, potential):
    complex_grid = grid[0] + grid[1]*1j
    n = len(complex_grid)
    
    w = np.zeros((n, n), dtype=np.complex128)
    
    # On the edge
    for i in range(n):
        w[i,-1] = (potential[i,-2] - potential[i,-1]) / (complex_grid[i,-2] - complex_grid[i,-1])
    
    # The rest of the grid
    for i in range(n):
        for j in range(n-1):
            w[i,j] = (potential[i,j+1] - potential[i,j]) / (complex_grid[i,j+1] - complex_grid[i,j])
    
    x_vel = w.real
    y_vel = -w.imag
    
    return x_vel, y_vel

X_transformed, Y_transformed = transform((X, Y))
transformed_x_vel, transformed_y_vel = compute_velocity((X_transformed, Y_transformed), optimal_potential)
transformed_pressure = -(transformed_x_vel**2 + transformed_y_vel**2)/2

x_vel, y_vel = compute_velocity((X, Y), optimal_potential, transformed=False)
pressure = -(x_vel**2 + y_vel**2)/2

plt.figure()
plt.contourf(X_transformed, Y_transformed, transformed_pressure, np.linspace(-1,0,60))
plt.plot(x_coords_transformed, y_coords_transformed, 'k-', linewidth=.7)
plt.axis('equal')
plt.colorbar()

plt.figure()
plt.contourf(X, Y, pressure, 50)
plt.axis('equal')
plt.colorbar()

#%% Exercise g
def contour_integral(grid, indices, pressure):
    n = len(indices)
    complex_grid = grid[0] + grid[1]*1j
    pressures = [pressure[indices[i][0], indices[i][1]] for i in range(n)]
    delta_z = [(complex_grid[indices[i+1][0], indices[i+1][1]] - complex_grid[indices[i-1][0], indices[i-1][1]])/2 if i+1<n else
               (complex_grid[indices[0][0], indices[0][1]] - complex_grid[indices[i-1][0], indices[i-1][1]])/2 for i in range(n)]
    print(complex_grid[indices[1][0],indices[-1][0]])
    return 1j*sum([pressures[i]*delta_z[i] for i in range(n)])

indices_1 = [(200,i) for i in range(500)]
indices_2 = [(300,i) for i in range(500)]

transformed_complex_grid = X_transformed + 1j*Y_transformed

# print(contour_integral((X, Y), indices_1, pressure))
# print(contour_integral((X, Y), indices_2, pressure))

# print(contour_integral((X_transformed, Y_transformed), indices_1, transformed_pressure))
# print(contour_integral((X_transformed, Y_transformed), indices_2, transformed_pressure))
contour_integral((X_transformed, Y_transformed), indices_1, transformed_pressure)
contour_integral((X_transformed, Y_transformed), indices_2, transformed_pressure)

