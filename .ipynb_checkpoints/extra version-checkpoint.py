import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # prints the whole array


def lin_func(x, a, b):
    """
    Linear function of x
    :param x: data to perform regression on
    :param a: slope param
    :param b: intercept param
    """
    
    return a*x + b
    
def exp_func(x, a, b):
    
    """
    Exponential function of x
    :param x: data to perform regression on
    :param a: coefficient
    :param b: coefficient
    """
    
    return b * (np.e ** (a*x))

background = np.loadtxt(fname = 'RadioactiveDecay_TuesdayOct2_2018_background.txt', delimiter = '\t', skiprows = 2) # reading data
decay = np.loadtxt(fname = 'RadioactiveDecay_TuesdayOct2_2018_decay.txt', delimiter = '\t', skiprows = 2)

delta_t = 20 # in this case, delta_t is 20 seconds

indices = (background[:,0] - 1) * delta_t # creating index array, converting samples to seconds
# using -1 because if 0 is not in the indices, curve_fit() breaks

rad_data = decay[:,1] - np.mean(background[:,1]) # subtracting mean background radiation from decay
rad_unc = np.sqrt(np.mean(background[:,1]) + decay[:,1]) # calculating uncertainty for each observation in decay file

# converting data to rates

rad_data_rates = rad_data/delta_t
rad_unc_rates = rad_unc/delta_t

# Generating data_line and data_exp using lin_func and exp_func:

data_params_line, _ = curve_fit(lin_func, indices, np.log(rad_data))
data_params_exp, _ = curve_fit(exp_func, indices, rad_data, p0 = (-1, 1))
print(data_params_line, data_params_exp)

data_line = lin_func(indices, data_params_line[0], data_params_line[1])
data_exp = exp_func(indices, data_params_exp[0], data_params_exp[1])


hl_line = abs((np.log(2)/data_params_line[0])) # finding half-life, using absolute value because line and exp function have negative slope params 
hl_exp = abs((np.log(2)/data_params_exp[0]))

print(hl_line, hl_exp) # printing estimated half-life for both linear and exponential function, in seconds

# Plotting log of number of counts against sample numbers

plt.figure(figsize = (10, 10))
plt.errorbar(indices, np.log(rad_data), xerr = None, yerr = (rad_unc/rad_data), fmt = 'o')
plt.plot(indices, data_line, color = 'r')
plt.xlabel('Sample Number'), plt.ylabel('Log of Number of Counts')

# Plotting number of counts against sample numbers

plt.figure(figsize = (10, 10))
plt.errorbar(indices, rad_data, xerr = None, yerr = rad_unc, fmt = 'o')
plt.plot(indices, data_exp, color = 'r')
plt.xlabel('Sample Number'), plt.ylabel('Number of Counts')

plt.show()
