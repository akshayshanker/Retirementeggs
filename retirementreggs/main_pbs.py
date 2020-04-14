import numpy as np
from numpy import genfromtxt
import csv
import numba 
from quantecon import tauchen

import time
from pathos.multiprocessing import ProcessingPool
import dill as pickle 

from egg_basket  import HousingModel, housingmodel_operator_factory, housingmodel_function_factory
from randparam 	 import rand_p_generator
from profiles_moments import genprofiles

import copy
import sys

"""
Housing model main

This code imports all parameter values, creates the utility and bequest functions
uses the parmaters to create an instance of the housiong model class and then solves
the model for housing and consumption policy functions
"""


# Read in and prepare parameter values 

# normalisation[0] is the value that scales the dollars in the model
# normalisation[1] is the scaling of the utility and bequest function (for machine precision)

# mental check: we can scale the utlity and bequest since maximisation is scale invariant. 
# the expected lifetime utility will be a sum of utilities and bequest functions, 
# hence we are scaling ifetime utility by a constant and optimal solution path will be constant '
# (but we also need to scale the adjustment cost parameters since they turn up in the utility. 
#

normalisation = np.array([1e5, 1e5])
ncores = int(48)
param_deterministic = {}
param_random_bounds = {}

with open('deterministic_params_{}.csv'.format(sys.argv[2]), newline='') as pscfile:
    reader = csv.DictReader(pscfile)
    for row in reader:
        param_deterministic[row['parameter']] = np.float64(row['value'])

with open('random_param_bounds.csv', newline='') as pscfile:
    reader_ran = csv.DictReader(pscfile)
    for row in reader_ran:
        param_random_bounds[row['parameter']] = np.float64([row['LB'],row['UB']])

lambdas  			= genfromtxt('lambdas_{}.csv'.format(sys.argv[3]), delimiter=',')[1:]
survival 			= genfromtxt('survival_probabilities_{}.csv'.format(sys.argv[3]), delimiter=',')[0:]
vol_cont_points 	= genfromtxt('vol_cont_points.csv', delimiter=',')[1:]
risk_share_points 	= genfromtxt('risk_share_points.csv', delimiter=',')[1:]
parameters 			= rand_p_generator(param_deterministic, param_random_bounds, deterministic = 1)



#Define functions 
functions = {}

functions['u'], functions['uc'], functions['uh'], functions['b'], \
functions['b_prime'], functions['y'], functions['DB_benefit'], \
functions['adj_p'], functions['adj_v'], functions['adj_pi'] = housingmodel_function_factory(parameters, lambdas, normalisation)



	
og = HousingModel(functions, parameters, survival, vol_cont_points, risk_share_points, ncores)   

gen_EVX, bellman_operator, a_prime_func = housingmodel_operator_factory(og)


og.solution 	= {}
og.accountdict 	= {}
og.accountdict[1] 	= 'DC'
og.accountdict[0] 	= 'DB'

t_end = og.tzero - 1


for account_type in [0,1]:
	EXV_bar = copy.copy(og.EVX_int) # set initial value for conditional value function 
	og.solution[og.accountdict[account_type]] = {}
	EXV_bar = EXV_bar.reshape(og.grid_size_A,og.grid_size_H, og.grid_size_Q)
	
	for i in range(int(og.T-t_end)):#NO MAGIC NUMBERS HERE!
		start_time = time.time()
		t = og.T - i
		print("Now solving value function for age %s "%(t))
		V_sol, rho_c_sol, rho_h_sol, a_prime_vals, h_inv = bellman_operator(t, EXV_bar, account_type) #Bellman operator returns uncondtioned policy functions and value function 
		end_time = time.time()
		print("Solved value function sans cond for age %s in %s seconds "%(t, end_time-start_time ))

		#take conditonal expectation of the value function on time t continuous state and labour shock !
		EXV_bar, PC_v, PC_pi = gen_EVX(V_sol,t, account_type)
		
		#transpose the value of the conditioned value function to each row corresponds to a labour shock
		EXV_bar_prime = np.transpose(EXV_bar)	

		#convert each row into a reshaped array that can be entered in the eval linear function

		if t> og.R:
			EXV_bar = EXV_bar_prime.reshape(og.grid_size_A,og.grid_size_H,og.grid_size_Q)
		else: 
			EXV_bar 	  = []		
			for  i in range(len(EXV_bar_prime)):
				EXV_bar.append(EXV_bar_prime[i].reshape(og.grid_size_A, og.grid_size_DC, og.grid_size_H,  og.grid_size_Q))

		og.solution[og.accountdict[account_type]][t] = [rho_c_sol, rho_h_sol, EXV_bar, PC_v, PC_pi]											
		end_time = time.time()
		print("Solved value function for age %s in %s seconds "%(t, end_time-start_time ))




#save results on Gadi scratch drive 

#pickle.dump(og, open("/scratch/pv33/{}.mod".format(sys.argv[1]),"wb"))

#Instead of saving og, generate time series and save time-series 

TSALL_10, TSALL_14 = genprofiles(og, N=100) 

pickle.dump(TSALL_10, open("/scratch/pv33/TSALL_10_{}.mod".format(sys.argv[1]),"wb"))
pickle.dump(TSALL_14, open("/scratch/pv33/TSALL_14_{}.mod".format(sys.argv[1]),"wb"))



