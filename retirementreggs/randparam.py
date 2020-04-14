import numpy as np

def rand_p_generator(
					param_deterministic, # dictionary containing deterministic parameters
					param_random_bounds, # dictionary containing random parameters 
					deterministic = 1,	 # flag to make all parmaeters determinstic 
					):

	"""Function generates list of parameters with 
	random parameters generated for params in the dictionary "pram_bounds"
	and deterministic parmameters for params in the dictionary "parameters". 

	Note: parameters in param_bounds will override any deterministic parameters in parameters
	If you want to make a parameter deterministic, remove it from the param_bounds list 
	"""

	parameters = {}

	# first pull out all the parameters that are deterministic

	for key in param_deterministic:
		parameters[key] = param_deterministic[key]

	if deterministic == 0:

		for key in param_random_bounds:
			parameters[key]  = np.random.uniform(param_random_bounds[key][0], param_random_bounds[key][1])

	return parameters