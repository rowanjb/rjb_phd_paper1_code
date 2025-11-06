# This complete script is straight from ChatGPT (and it works perfectly!)
# You can use it to parametrically define exponentially-increasing cell thicknesses for a model of the form f(x)=A⋅e**(B⋅x)+C 
# Where you can specify 
#       x1          the first cell index, which should be 1 (i.e., the surface cell)
#       x2          the last cell index, e.g., 50 (i.e., the bottom cell)
#       f(x1)       the depth of the bottom of the first cell (e.g., 1 m)
#       f(x2)       the depth of the bottom of the last cell (e.g., 500 m)
#       min_slope   the minimum slope between x1 and x2
# By setting the min_slope, you can ensure that there are no super-thin cells that cause the CFL criterion to require a small dt

from scipy.optimize import fsolve
import numpy as np

# Function to solve the system of equations
def equations(vars, x1, x2, fx1, fx2, min_slope):
    A, B, C = vars
    
    # Equation 1: f(x1) = fx1
    eq1 = A * np.exp(B * x1) + C - fx1  # f(x1) = fx1
    # Equation 2: f(x2) = fx2
    eq2 = A * np.exp(B * x2) + C - fx2  # f(x2) = fx2
    # Equation 3: f'(x1) = min_slope (derivative condition at x = x1)
    eq3 = A * B * np.exp(B * x1) - min_slope  # f'(x1) = min_slope
    return [eq1, eq2, eq3]

# Function to check the derivative at a point
def derivative_condition(A, B, x):
    return A * B * np.exp(B * x)

# Generalized function to find the parameters A, B, and C
def find_parameters(x1, x2, fx1, fx2, min_slope):
    # Initial guess for A, B, C
    initial_guess = [1, 0.1, 1]
    
    # Solve the system
    A, B, C = fsolve(equations, initial_guess, args=(x1, x2, fx1, fx2, min_slope))
    
    # Check the derivative condition at x = x1
    slope_at_x1 = derivative_condition(A, B, x1)
    # Check the derivative condition at x = x2
    slope_at_x2 = derivative_condition(A, B, x2)
    
    # Return the results
    return A, B, C, slope_at_x1, slope_at_x2

# Adding this function manually to return a list of cell thicknesses, to be called elsewhere
def return_cell_thicknesses(x1, x2, fx2, A, B, C):
    ids = range(x1,x2+1) # Ids of the cells (should range from e.g., 1 to 50)
    z = [A*np.exp(B*id)+C for id in ids] # list of depths /at the bottom of each cell/ 
    # thicknesses calculated with np.diff and rounded to be integers. Prepend=0 ensures another cell at start to that len is correct
    dz = [int(np.round(dz_float)) for dz_float in np.diff(z,prepend=0)] 
    dfx2 = np.sum(dz) - fx2 # Seeing how far off we are from the target fx2 after rounding and converting to ints
    dz[-1] = dz[-1] - dfx2 # Adjusting the bottom cell so that the total domain is equal to fx2
    return dz

if __name__=="__main__":
    # Example usage:
    x1 = 1        # x1 value
    x2 = 50       # x2 value
    fx1 = 1       # f(x1)
    fx2 = 500     # f(x2)
    min_slope = 1 # Minimum slope (should probably > x1)

    A, B, C, slope_at_x1, slope_at_x2 = find_parameters(x1, x2, fx1, fx2, min_slope)

    # Print the results
    print(f"A = {A}, B = {B}, C = {C}")
    print(f"Slope at x = {x1}: {slope_at_x1}")
    print(f"Slope at x = {x2}: {slope_at_x2}")

    dz = return_cell_thicknesses(x1, x2, fx2, A, B, C)
    print(len(dz))
    print(np.sum(dz))
    print(dz)


