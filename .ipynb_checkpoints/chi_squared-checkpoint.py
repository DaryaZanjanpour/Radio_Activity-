import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf) # prints the whole array
from statistics import stdev 

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
indices =np.arange(0,1200,20)


rad_data = decay[:,1] - np.mean(background[:,1]) # subtracting mean background radiation from decay
rad_unc = np.sqrt(np.mean(background[:,1]) + decay[:,1]) # calculating uncertainty for each observation in decay file
                 


def chi_squared_linear(dof, x, y, sigma, a, b):
    """
    Returns chi squared of a linear model
    
    :param: dof, degrees of freedom of the data
    :param: x, independent data set
    :param: y, dependent data set
    :param: sigma, measurement error in y
    :param: a, slope parameter in model
    :param: b, intercept parameter in model
    """
    
    chi_squared = (1/dof) * np.sum(((y - lin_func(x, a, b))/sigma)**2)
    
    return chi_squared


delta_t = 20 # in this case, delta_t is 20 seconds

# converting data to rates

rad_data_rates = rad_data/delta_t
rad_unc_rates = rad_unc/delta_t

# Generating data_line and data_exp using lin_func and exp_func:

data_params_line, _ = curve_fit(lin_func, indices, np.log(rad_data))
data_params_exp, _ = curve_fit(exp_func, indices, rad_data, p0 = (-1, 1))
print(data_params_exp)

data_line = lin_func(indices,data_params_line[0],data_params_line[1])
data_exp = exp_func(indices,data_params_exp[0], data_params_exp[1])

print(data_exp)

hl_line = np.log(2)/data_params_line[0]
hl_exp = np.log(2)/data_params_exp[0]

print("uncertainty of linear data", stdev(data_line))
print("uncertainty of expotential data", np.average(np.absolute(stdev(data_exp)/data_exp)))
print("HALF LIFE=", hl_line, hl_exp)

# Plotting log of number of counts against sample numbers

plt.figure(figsize = (10, 10))
plt.title("Linear regression of on Sample number and Log of Number of Counts")
plt.errorbar(indices, np.log(rad_data), xerr = None, yerr = (rad_unc/rad_data), fmt = 'o')
plt.plot(indices, data_line, color = 'r')
plt.xlabel('Time (s)'), plt.ylabel('Log of Number of Counts')
plt.savefig('1')

# Plotting number of counts against sample numbers

plt.figure(figsize = (10, 10))
plt.title("Exponential  regression")
plt.errorbar(indices, rad_data, xerr = None, yerr = rad_unc, fmt = 'o')
plt.plot(indices, data_exp, color = 'r')
plt.xlabel('time (s)'), plt.ylabel('Number of Counts')
plt.savefig('2')
plt.show()




chi_squared_linear_function = chi_squared_linear(58, indices, np.log(rad_data), rad_unc/rad_data, data_params_line[0], data_params_line[1])
print("chi squared linear=",chi_squared_linear_function)

chi_squared_exp = chi_squared_linear(58, indices, rad_data,  rad_unc, data_params_exp[0], data_params_exp[1])
print("chi squared exponential=",np.log(chi_squared_exp))