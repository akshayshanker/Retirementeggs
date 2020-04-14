
"""
This module contains the functions to generate time series of agents
and moments from a solved HousingModel 
 
Functions: genprofiles
            Generates the a time series of N agents from the HousingModel class 
           
           housingmodel_function_factory
            Generates moments from time-series 

"""

import numpy as np
from quantecon import tauchen
import matplotlib.pyplot as plt

from itertools import product
from sklearn.utils.extmath import cartesian
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
import pandas as pd

import matplotlib.pyplot as plt
import copy
from egg_basket  import HousingModel, housingmodel_operator_factory, \
						housingmodel_function_factory
import dill as pickle 

import os
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib.ticker as ticker


from itertools import combinations, permutations

import seaborn as sns



def genprofiles(og,
				N = 100,
				norm = 100000):

	

	"""For a solved housing model, returns a time-series of profiles 

	(currently males only) 

	Parameters
	----------
	og: class
		Housing nmodel  
	length: int
		time-series length (end age determined by og.T and og.tzero)
	N: int
		Number of agents to simulate 

	Returns
	-------
	TSALL
		Dictionary containing time-series
	"""
	
	# unpack and define functions to be used in time-series generator 
	def a_prime_func_gen(c,                     # Current period consumption (C_{t}) choice
	                      h,            # Current period housing (H_{t}) choice 
	                      x_disc_vals,       # Current discrete state. Set to 0 if t>R. This an index of the discrete cartesian grid  (np.arange(len(X_disc))) NOT the val. 
	                      x_cont_vals,       # Current continuous state. This is an index in the nodes grids generated from the continuous grid object 
	                      t,            # Current age
	                      account_type, # O for DB and 1 for DC
	                      wage,         # wage received this period 
	                      DB_payout,    # DB payout received this period (mostly 0)
	                      ):

		if t==og.R:
		    # This gives a tuple of indices for each of the discrete state spaces! not individual state space values!

		    # This gives the discrete state values                     
		    #x_disc_vals = X_disc_vals[x_disc]               
		    
		    # Remember the indexing housing brought *in from the previous period* is x_disc_vals[2]
		    # No depreciation of x_disc_vals[2]!!! Depreciation does enter into x_prime. 


		    # value of \tau*\Chi_{housing invesmtnet made in current period}
		    # At age R, the DC balance matters so we are using the worker continuous grid. House value is indexed at 2
		    if h == x_cont_vals[2]:                        
		        tau_star = 0
		    
		    else: 
		        tau_star   = og.tau_housing

		    # next liquid period assets 

		    a_prime    =  (1+og.r)*x_cont_vals[0] - c - x_cont_vals[3]*(h -x_cont_vals[2])\
		                    - tau_star*x_cont_vals[3]*h\
		                    + x_cont_vals[1] + DB_payout*(1-account_type)

		if t>og.R:


		    # Note that the retirement age is R so the final
		    # period when next period consumption will be on
		    # the worker grid is R (even thought they do not recieve wage at R, they receive dc pay and DB)
		    

		    if h == x_cont_vals[1]:
		        tau_star = 0
		    else: 
		        tau_star   = og.tau_housing

		    a_prime    =  (1+og.r)*x_cont_vals[0] - c -x_cont_vals[2]*(h -x_cont_vals[1])\
		                    - tau_star*x_cont_vals[2]*h

		   # print(x_cont_vals)
		   # print(x_cont_vals[2])


		if t<og.R:


			# Remember the indexing housing brought *in from the previous period* is x_disc_vals[2]. 
			# No depreciation of x_disc_vals[2]!!! Depreciation does enter into x_prime. 

			#x_disc_vals = X_disc_vals[x_disc]

			#print(x_cont_vals)

			v_cont_rate = x_disc_vals[3]

			                                                                
			if h == x_cont_vals[2]:
			    tau_star = 0

			else: 
			    tau_star         = og.tau_housing

			# Next period liquid capital 

			a_prime         =   (1+og.r)*x_cont_vals[0] - c - x_cont_vals[3]*(h -x_cont_vals[2])\
			                    - tau_star*x_cont_vals[3]*h\
			                    + (1-v_cont_rate -og.v_S -og.v_E)*wage                      

		return a_prime

	def closest_node(node, nodes):
	    nodes 	= np.asarray(nodes)
	    dist_2 	= np.sum((nodes - node)**2, axis=1)
	    return np.argmin(dist_2)


	def seriesgen(age, wave_length):
		""" Returns a time series of one agent to age in baseline year and baseline year + wave_lenght
		"""
		og.tzero 	= int(og.tzero)
		length = int(age + wave_length+2) 

		# generate sequences of shocks for this agent i 
		# now remeber that the start year is og.tzero. 
		# we generate shocks such that TS_A[t] gives the value of
		# the shock at t

		TS_A  	= np.zeros(length)
		TS_H  	= np.zeros(length)
		TS_DC 	= np.zeros(length)
		TS_C  	= np.zeros(length)
		TS_V  	= np.zeros(length)
		TS_PI 	= np.zeros(length)
		TS_wage = np.zeros(length)
		TS_hinv = np.zeros(length)
		adj_V 	= np.zeros(length)
		adj_pi 	= np.zeros(length)
		P_h		= np.zeros(length)


		# generate sequence of house price shocks

		P_h[og.tzero]  = 1/((1+og.r_H)**(age - og.tzero))

		for t in np.arange(og.tzero, len(P_h)-1):
			P_h[t+1] = (1+og.r_H)*P_h[t]  

		# initialize continuous points 

		TS_A[og.tzero] = 1e-5
		TS_H[og.tzero] = 1e-5
		TS_DC[og.tzero]= 1e-5

		# generate time series of wage shocks, beta_hat and alpha_hat draws for this guy

		W= 	og.labour_mc.simulate(ts_length = length)
		beta_hat_ts =	og.beta_mc.simulate(ts_length = length)
		alpha_hat_ts= 	og.alpha_mc.simulate(ts_length = length)

		DC_H= 	np.exp(og.r_h + og.lnrh_mc.simulate(ts_length = length)) # these need to be made deterministic 
		DC_L = 	np.exp(og.r_l + og.lnrl_mc.simulate(ts_length = length)) # these need to be made deterministic 	


		# get the initial value of the series and draw from account type
		

		#og.sigma_plan =.03
		disc_exog_ind = int(np.where((np.where(W[int(og.tzero)]== og.E)[0] \
								== og.X_disc_exog[:,0]) \
							&   (og.X_disc_exog[:,1] == np.where(og.alpha_hat \
								 == alpha_hat_ts[int(og.tzero)])[0]) \
							& 	(og.X_disc_exog[:,2] == np.where(og.beta_hat\
								 == beta_hat_ts[int(og.tzero)])[0]))[0])


		V_DC 		=  eval_linear(og.X_cont_W, \
				 		og.solution['DC'][og.tzero][2][disc_exog_ind].\
				 			reshape(og.grid_size_A,og.grid_size_DC,og.grid_size_H, og.grid_size_Q), \
				 		np.array([TS_A[0],TS_DC[0],TS_H[0], P_h[0]]))

		V_DB 		= eval_linear(og.X_cont_W, \
		 				og.solution['DB'][og.tzero][2][disc_exog_ind].\
		 				reshape(og.grid_size_A,og.grid_size_DC,og.grid_size_H, og.grid_size_Q), \
		 				np.array([TS_A[0],TS_DC[0],TS_H[0], P_h[0]])) 

		Prob_DC 	= np.exp((V_DC - og.adj_p(og.tzero))/(og.sigma_plan))/(
						np.exp((V_DC - og.adj_p(og.tzero))/(og.sigma_plan)) \
					+   np.exp((V_DB )/(og.sigma_plan) ) )    

		account_ind 	= np.random.binomial(n=1, p = Prob_DC)
		account_type 	= og.accountdict[account_ind]


		for t in range(int(og.tzero), age+ wave_length+1):
			if t <= og.R:
			#get the index for the labour shock
				E_ind = int(np.where(W[t]==og.E)[0])
				beta_ind 	= int(np.where(beta_hat_ts[t]== og.beta_hat)[0])
				alpha_ind 	= int(np.where(alpha_hat_ts[t]== og.alpha_hat)[0])

				#get continuous state index
				cont_ind = closest_node(np.array([TS_A[t], np.array(TS_DC[t])\
								,np.array(TS_H[t]), np.array(P_h[t])]), og.X_W_contgp)

				#pull out the probability of voluntary contribution probability matrix for this cont state

				probs_v = og.solution[account_type][t][3][int(cont_ind)]

				#pull out the PMF of making the voluntary contributions for this continuous index 
				v_prob_PMF	= probs_v[np.where((probs_v[:,1].astype(int)==E_ind)\
								& (probs_v[:,2].astype(int) == alpha_ind)\
								& (probs_v[:,3].astype(int) == beta_ind) ) ][:,4:6]

				#pick a random draw for the voluntart contribution (index in the vol.cont grid)
				V_ind = int(np.random.choice(v_prob_PMF[:,0], 1, p=v_prob_PMF[:,1]))

				#pull our probabiltiy of risk share matrix for this cont state 
				probs_pi = og.solution[account_type][t][4][cont_ind]

				#pull out PMF from matrix 
				pi_prob_PMF =  probs_pi[np.where((probs_pi[:,1].astype(int)==E_ind ) \
								& (probs_pi[:,2].astype(int) == alpha_ind) \
								& (probs_pi[:,3].astype(int) == beta_ind) \
								& (probs_pi[:,4].astype(int)== V_ind ))][:,5:7]

				#draw of risk choice 
				pi_ind     	= int(np.random.choice(pi_prob_PMF[:,0], 1,\
									 p=pi_prob_PMF[:,1]))

				#calculate age for agent 
				TS_wage[t]	= og.y(t,og.E[E_ind])

				#now we can find the discrete index 
				x_disc 		= int(np.where((og.X_disc[:,0].astype(int) == E_ind)\
								 & (og.X_disc[:,1].astype(int) == alpha_ind) \
								 & (og.X_disc[:,2].astype(int) == beta_ind) \
								 & (og.X_disc[:,3].astype(int) == V_ind) \
								 & (og.X_disc[:,4].astype(int) == pi_ind))[0])

				#consumption this period 
				TS_C[t]  	= eval_linear(og.X_cont_W, \
								og.solution[account_type][t][0][x_disc].\
									reshape(og.grid_size_A,og.grid_size_DC,og.grid_size_H,  og.grid_size_Q), \
								np.array([TS_A[t],TS_DC[t],TS_H[t], P_h[t]]))

				#housing taken in to the next period 
				TS_H[t+1]  = np.min([og.H_max,np.max([og.H_min,eval_linear(og.X_cont_W,\
								og.solution[account_type][t][1][x_disc].\
									reshape(og.grid_size_A,og.grid_size_DC,og.grid_size_H,  og.grid_size_Q),\
								np.array([TS_A[t],TS_DC[t],TS_H[t], P_h[t]]))])])

				TS_hinv[t] = TS_H[t+1] -TS_H[t]
				

				if t== og.R:
					DB_payout = og.DB_benefit(t, t-og.tzero, og.y(t,og.X_disc_vals[x_disc][0]), \
									og.X_disc[x_disc][0], og.P_E,og.P_stat,og.E)
				else:
					DB_payout = 0

				#liquid assets next period 		
				x_cont_vals = np.array([TS_A[t], TS_DC[t], TS_H[t], P_h[t]])	

				X_disc_vals = np.array(og.X_disc_vals[x_disc])

				TS_A[t+1]  	= np.min([og.A_max_W, np.max([og.A_min, a_prime_func_gen(TS_C[t],TS_H[t+1], \
							   X_disc_vals, x_cont_vals, t, account_ind, TS_wage[t], DB_payout)])])


				TS_DC[t+1] = (1+ (1-og.Pi[pi_ind])*DC_L[t]/100 \
                                + og.Pi[pi_ind]*DC_H[t]/100)*(TS_DC[t] \
                                + (account_ind*og.v_S + account_ind*og.v_E+ og.V[V_ind])*TS_wage[t])           

				#CHECK AGAIN THAT THE ABOVE ACTUALLY MAKES SESNSE: I.E. WE REALISE THE DC ASSETT SHOCK THIS PERIOD 

				TS_V[t] 	= og.V[V_ind]
				TS_PI[t] 	= og.Pi[pi_ind]

				adj_V[t]  		= og.adj_v(t, og.X_W_contgp[cont_ind][0])
				adj_pi[t]  		= og.adj_pi(t, og.X_W_contgp[cont_ind][1], og.adj_v(t, og.X_W_contgp[cont_ind][0]))

			if t > og.R:
				x_disc 		= 0 # set this as dummy
				x_cont_vals = np.array([TS_A[t], TS_H[t],  P_h[t]])	
				TS_C[t]  	= eval_linear(og.X_cont_R, og.solution[account_type][t][0].\
									reshape(og.grid_size_A,og.grid_size_H,  og.grid_size_Q), \
									np.array([TS_A[t],TS_H[t], P_h[t]]))
				TS_A[t+1]  	= np.min([og.A_max_R,np.max([0,a_prime_func_gen(TS_C[t], \
										TS_H[t+1],  x_disc, x_cont_vals, t, account_ind, \
											 0,0)])])
				TS_H[t+1] 	= np.min([og.H_max, np.max([og.H_min, \
								eval_linear(og.X_cont_R, \
								og.solution[account_type][t][1]. \
								reshape(og.grid_size_A,og.grid_size_H,  og.grid_size_Q), \
								 np.array([TS_A[t],TS_H[t], P_h[t]]))])])

				TS_hinv[t] = TS_H[t+1] -TS_H[t]
		
		w10 	=  np.array([account_ind,age, TS_A[age]*norm,\
							TS_H[age]*norm, TS_DC[age]*norm, TS_C[age]*norm, \
							TS_wage[age]*norm, TS_V[age], TS_V[age]*TS_wage[age]*norm, TS_PI[age], \
							int(TS_hinv[age]>0), int(TS_PI[age]>.7), int(TS_V[age]>0), \
							alpha_hat_ts[age], beta_hat_ts[age],adj_V[age], adj_pi[age]])

		
		age_wave_10 = age
		age 	= age+ wave_length

		# we denote the age at wave_14 by thier age at 10 so they go in 2010 bucket

		w14 	= np.array([account_ind,age_wave_10, TS_A[age]*norm,\
							TS_H[age]*norm, TS_DC[age]*norm, TS_C[age]*norm, \
							TS_wage[age]*norm, TS_V[age],TS_V[age]*TS_wage[age]*norm, TS_PI[age], \
							int(TS_hinv[age]>0), int(TS_PI[age]>.7), int(TS_V[age]>0), \
							alpha_hat_ts[age], beta_hat_ts[age],adj_V[age], adj_pi[age]])
		
		return [w10, w14]

	#series 		= p.map(lambda i: seriesgen(i, DC_H, DC_L), np.arange(10))

	TSALL_10 = []
	TSALL_14 = []

	for age in np.arange(19, 65):
		for i in range(N):
			x = seriesgen(age, 4)
			TSALL_10.append(x[0])
			TSALL_14.append(x[1])

	TSALL_10_df = pd.DataFrame(TSALL_10)
	TSALL_14_df = pd.DataFrame(TSALL_14)

	col_list = list(['account_type', \
						  'Age', \
						  'wealth_fin', \
						  'wealth_real', \
						  'super_balance', \
						  'consumption', \
						  'Wages',\
						  'vol_cont', \
						  'vol_total',\
						  'risky_share', \
						  'house_adj', \
						  'risky_share_adj',\
						  'vol_cont_adj', \
						  'alpha_hat', \
						  'Beta_hat', \
						  'Adjustment_cost_v', \
						  'Adjustment_cost_pi'])


	TSALL_10_df.columns = col_list
	TSALL_14_df.columns = col_list
	
	return TSALL_10_df, TSALL_14_df



def gen_moments(TSALL_10_df, TSALL_14_df):

	age = np.arange(18, 65)
	main_df = pd.DataFrame(age)

	# age buckets are LHS closed RHS open
	# final age bucket is t = (58, 63] and hence age = (59, 64]
	# first age bucket is age = (19, 24] 
	age_buckets = np.arange(19, 65,5)

	
	keys_vars = set(TSALL_10_df.keys())
	excludes = set(['Adjustment_cost_v', \
				 'Adjustment_cost_pi', 'alpha_hat', 'Beta_hat'])

	TSALL_10_df.drop(excludes, axis = 1)    
	

	"""	for  key in keys_vars.difference(excludes):
			#print(key)
			main_df = pd.concat([main_df, pd.DataFrame( \
						np.transpose(TSALL[key][:,18:65]*100)). \
						add_prefix('{}'.format(key))], axis =1)


		# generate moments for wave 10
		main_df['id'] = main_df.index
		main_df = main_df.rename(columns ={0:'Age'})
		main_df = pd.wide_to_long(main_df, stubnames = \
					 list(keys_vars.difference(excludes)), i = "id", j = "memberid")   
	"""
	main_df = TSALL_10_df
	main_df['vol_cont_adj']= main_df['vol_cont_adj']
	main_df['risky_share_adj']= main_df['risky_share_adj']
	main_df['risky_share']= main_df['risky_share']
	main_df['vol_cont']=main_df['vol_cont']

	# adjust age so it is 'real age'

	main_df['Age'] += 1    

	#account_type_df  = pd.DataFrame(TSALL['account_type_all'])     
	#account_type_df['memberid'] = account_type_df.index 
	#account_type_df['memberid'].reset_index()
	#main_df = pd.merge(main_df, account_type_df, left_on = 'memberid', \
	#			 right_on = 'memberid', how = "left", validate = "m:1")  
	#main_df = main_df.rename(columns = {0: 'account_type'})

	means = main_df.groupby(pd.cut(main_df['Age'], age_buckets)).mean().add_prefix('mean_')  
	means = means.reset_index()   

	sds_all = main_df.groupby(pd.cut(main_df['Age'], \
				age_buckets)).std().add_prefix('sd_')
	sds_all = sds_all.reset_index().drop(['Age','sd_Age', \
				'sd_account_type'], axis =1)    

	sds_DB 	= main_df[main_df['account_type']==0].groupby(pd.cut \
				(main_df[main_df['account_type']==0]['Age'], age_buckets)) \
				.std().add_prefix('sd_').add_suffix('DB')
	sds_DB	= sds_DB.reset_index().drop(['Age','sd_AgeDB', \
				'sd_account_typeDB'], axis =1)

	sds_DC 	= main_df[main_df['account_type']==1].groupby(pd. \
				cut(main_df[main_df['account_type']==1]['Age'], age_buckets)). \
				std().add_prefix('sd_').add_suffix('DC')
	sds_DC  = sds_DC.reset_index().drop(['Age','sd_AgeDC', \
				 'sd_account_typeDC'], axis =1)

	corrlist = list(list(permutations(keys_vars.difference(excludes),2)))
	corrs_df = pd.DataFrame(means.index)  

	for corrs in corrlist:

		corrs_temp  = main_df.groupby(pd.cut(main_df['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp  = corrs_temp.reset_index()
		corrs_temp  = np.array(corrs_temp[corrs_temp.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)], \
					 		axis=1, inplace=False)

		# for DB only
		corrs_temp_DB  = main_df[main_df['account_type']==0] \
						 .groupby(pd.cut(main_df[main_df['account_type']==0]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DB  = corrs_temp_DB.reset_index()
		corrs_temp_DB  = np.array(corrs_temp_DB[corrs_temp_DB.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DB)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DB'], \
					 		axis=1, inplace=False)

		# for DC only
		corrs_temp_DC  = main_df[main_df['account_type']==1] \
						 .groupby(pd.cut(main_df[main_df['account_type']==1]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DC  = corrs_temp_DC.reset_index()
		corrs_temp_DC  = np.array(corrs_temp_DC[corrs_temp_DC.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DC)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DC'], \
					 		axis=1, inplace=False)


			
				
	corrs_df = corrs_df.add_prefix('corr_')

	moments = pd.concat([means.reset_index(), sds_all.reset_index(),\
				 corrs_df.reset_index(), sds_DB.reset_index()\
				 ,sds_DC.reset_index()  ], axis = 1)   
	
	moments_10 = moments.drop(['index'],axis = 1).add_suffix('_wave10')   

	main_df10 = main_df

	main_df10 = main_df10.add_suffix('_wave10')

	# now generate moments for wave 14

	# this method will need to change depending on meeting with ISA 

	main_df = TSALL_14_df
	main_df['vol_cont_adj']= main_df['vol_cont_adj']
	main_df['risky_share_adj']= main_df['risky_share_adj']
	main_df['risky_share']= main_df['risky_share']
	main_df['vol_cont']=main_df['vol_cont']

	# adjust age so it is 'real age'

	main_df['Age'] += 1    

	means = main_df.groupby(pd.cut(main_df['Age'], age_buckets)).mean().add_prefix('mean_')  
	means = means.reset_index() 

	sds_all = main_df.groupby(pd.cut(main_df['Age'], \
				age_buckets)).std().add_prefix('sd_')
	sds_all = sds_all.reset_index().drop(['Age','sd_Age', \
				'sd_account_type'], axis =1)    

	sds_DB 	= main_df[main_df['account_type']==0].groupby(pd.cut \
				(main_df[main_df['account_type']==0]['Age'], age_buckets))\
				.std().add_prefix('sd_').add_suffix('DB')
	sds_DB	= sds_DB.reset_index().drop(['Age','sd_AgeDB',\
				'sd_account_typeDB'], axis =1)

	sds_DC 	= main_df[main_df['account_type']==1].groupby(pd. \
				cut(main_df[main_df['account_type']==1]['Age'], age_buckets)). \
				std().add_prefix('sd_').add_suffix('DC')
	sds_DC  = sds_DC.reset_index().drop(['Age','sd_AgeDC',\
				 'sd_account_typeDC'], axis =1)

	corrlist = list(list(permutations(keys_vars.difference(excludes),2)))
	corrs_df = pd.DataFrame(means.index)  

	for corrs in corrlist:

		corrs_temp  = main_df.groupby(pd.cut(main_df['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp  = corrs_temp.reset_index()
		corrs_temp  = np.array(corrs_temp[corrs_temp.columns[1]] \
						.reset_index())[:,1]
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)], \
					 		axis=1, inplace=False)
		# for DB only
		corrs_temp_DB  = main_df[main_df['account_type']==0] \
						 .groupby(pd.cut(main_df[main_df['account_type']==0]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DB  = corrs_temp_DB.reset_index()
		corrs_temp_DB  = np.array(corrs_temp_DB[corrs_temp_DB.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DB)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DB'], \
					 		axis=1, inplace=False)

		# for DC only
		corrs_temp_DC  = main_df[main_df['account_type']==1] \
						 .groupby(pd.cut(main_df[main_df['account_type']==1]['Age'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		corrs_temp_DC  = corrs_temp_DC.reset_index()
		corrs_temp_DC  = np.array(corrs_temp_DC[corrs_temp_DC.columns[1]] \
						.reset_index())[:,1]
		
		corrs_df 	= pd.concat([corrs_df.reset_index(drop = True), \
						pd.DataFrame(corrs_temp_DC)  ], axis = 1) 
		corrs_df	=	corrs_df.set_axis([*corrs_df.columns[:-1], '_'.join(corrs)+'_DC'], \
					 		axis=1, inplace=False)

			
				
	corrs_df 		= corrs_df.add_prefix('corr_')

	moments_14 		= pd.concat([means.reset_index(), sds_all.reset_index(),\
				 		corrs_df.reset_index(), sds_DB.reset_index()\
				 		,sds_DC.reset_index()  ], axis = 1) \
						.reset_index() \
						.add_suffix('_wave14') \
						.drop(['Age_wave14', 'index_wave14', 'level_0_wave14'],
							 axis= 1)

	main_df14 		= main_df
	main_df14		= main_df14.add_suffix('_wave14')

	main_horiz 		= pd.concat((main_df10, main_df14), axis =1)  
	
	# auto-correlation
	auto_corrs_df = pd.DataFrame(means.index)

	for keys in keys_vars.difference(excludes):
		corrs = list((keys+'_wave10', keys+'_wave14'))
		autocorrs_temp 	= main_horiz.groupby(pd.cut(main_horiz['Age_wave10'], \
						 age_buckets))[corrs].corr().unstack().iloc[:,1]
		autocorrs_temp 	= autocorrs_temp.reset_index()
		autocorrs_temp  = np.array(autocorrs_temp[autocorrs_temp.columns[1]] \
						.reset_index())[:,1]
		auto_corrs_df 	= pd.concat([auto_corrs_df.reset_index(drop = True), \
							pd.DataFrame(autocorrs_temp)  ], axis = 1) 
		auto_corrs_df	=	auto_corrs_df.set_axis([*auto_corrs_df.columns[:-1], \
							 keys+'_autocorr'], \
			 				 axis=1, inplace=False)

	# generate conditional voluntary contribution 

	moments_10['mean_vol_cont_c_wave10']= moments_10['mean_vol_total_wave10']/moments_10['mean_vol_cont_adj_wave10']
	moments_14['mean_vol_cont_c_wave14']= moments_14['mean_vol_total_wave14']/moments_14['mean_vol_cont_adj_wave14']

	return pd.concat([moments_10, moments_14, auto_corrs_df], axis =1)


def sortmoments(moments_male, moments_female):
 	
 	empty = pd.DataFrame(np.zeros(9))

 	moments_sorted = pd.concat(
						[

						moments_male['sd_consumption_wave14_male'],
						moments_female['sd_consumption_wave14_female'],
						moments_male['sd_consumption_wave10_male'],
						moments_female['sd_consumption_wave10_female'],
						moments_male['mean_account_type_wave14_male'],
						moments_female['mean_account_type_wave14_female'],
						moments_male['mean_account_type_wave10_male'],
						moments_female['mean_account_type_wave10_female'],
						moments_male['corr_super_balance_risky_share_adj_DC_wave14_male'],
						moments_male['corr_super_balance_risky_share_adj_DB_wave14_male'],
						moments_female['corr_super_balance_risky_share_adj_DC_wave14_female'],
						moments_female['corr_super_balance_risky_share_adj_DB_wave14_female'],
						moments_male['corr_super_balance_risky_share_adj_DC_wave10_male'],
						moments_male['corr_super_balance_risky_share_adj_DB_wave10_male'],
						moments_female['corr_super_balance_risky_share_adj_DC_wave10_female'],
						moments_female['corr_super_balance_risky_share_adj_DB_wave10_female'],
						moments_male['corr_super_balance_risky_share_DC_wave14_male'],
						moments_male['corr_super_balance_risky_share_DB_wave14_male'],
						moments_female['corr_super_balance_risky_share_DC_wave14_female'],
						moments_female['corr_super_balance_risky_share_DB_wave14_female'],
						moments_male['corr_super_balance_risky_share_DC_wave10_male'],
						moments_male['corr_super_balance_risky_share_DB_wave10_male'],
						moments_female['corr_super_balance_risky_share_DC_wave10_female'],
						moments_female['corr_super_balance_risky_share_DB_wave10_female'],
						moments_male['corr_super_balance_vol_cont_adj_DC_wave14_male'],
						moments_male['corr_super_balance_vol_cont_adj_DB_wave14_male'],
						moments_female['corr_super_balance_vol_cont_adj_DC_wave14_female'],
						moments_female['corr_super_balance_vol_cont_adj_DB_wave14_female'],
						moments_male['corr_super_balance_vol_cont_adj_DC_wave10_male'],
						moments_male['corr_super_balance_vol_cont_adj_DB_wave10_male'],
						moments_female['corr_super_balance_vol_cont_adj_DC_wave10_female'],
						moments_female['corr_super_balance_vol_cont_adj_DB_wave10_female'],
						moments_male['corr_vol_total_super_balance_DC_wave14_male'],
						moments_male['corr_vol_total_super_balance_DB_wave14_male'],
						moments_female['corr_vol_total_super_balance_DC_wave14_female'],
						moments_female['corr_vol_total_super_balance_DB_wave14_female'],
						moments_male['corr_vol_total_super_balance_DC_wave10_male'],
						moments_male['corr_vol_total_super_balance_DB_wave10_male'],
						moments_female['corr_vol_total_super_balance_DC_wave10_female'],
						moments_female['corr_vol_total_super_balance_DB_wave10_female'],
						moments_male['corr_consumption_wealth_real_wave14_male'],
						moments_female['corr_consumption_wealth_real_wave14_female'],
						moments_male['corr_consumption_wealth_real_wave10_male'],
						moments_female['corr_consumption_wealth_real_wave10_female'],
						moments_male['wealth_real_autocorr_male'],
						moments_female['wealth_real_autocorr_female'],
						moments_male['wealth_fin_autocorr_male'],
						moments_female['wealth_fin_autocorr_female'],
						moments_male['consumption_autocorr_male'],
						moments_female['consumption_autocorr_female'],
						moments_male['sd_risky_shareDC_wave14_male'],
						moments_male['sd_risky_shareDB_wave14_male'],
						moments_female['sd_risky_shareDC_wave14_female'],
						moments_female['sd_risky_shareDB_wave14_female'],
						moments_male['sd_risky_shareDC_wave10_male'],
						moments_male['sd_risky_shareDB_wave10_male'],
						moments_female['sd_risky_shareDC_wave10_female'],
						moments_female['sd_risky_shareDB_wave10_female'],
						moments_male['sd_vol_totalDC_wave14_male'],
						moments_male['sd_vol_totalDB_wave14_male'],
						moments_female['sd_vol_totalDC_wave14_female'],
						moments_female['sd_vol_totalDB_wave14_female'],
						moments_male['sd_vol_totalDC_wave10_male'],
						moments_male['sd_vol_totalDB_wave10_male'],
						moments_female['sd_vol_totalDC_wave10_female'],
						moments_female['sd_vol_totalDB_wave10_female'],
						moments_male['sd_super_balanceDC_wave14_male'],
						moments_male['sd_super_balanceDB_wave14_male'],
						moments_female['sd_super_balanceDC_wave14_female'],
						moments_female['sd_super_balanceDB_wave14_female'],
						moments_male['sd_super_balanceDC_wave10_male'],
						moments_male['sd_super_balanceDB_wave10_male'],
						moments_female['sd_super_balanceDC_wave10_female'],
						moments_female['sd_super_balanceDB_wave10_female'],
						moments_male['sd_wealth_real_wave14_male'],
						moments_female['sd_wealth_real_wave14_female'],
						moments_male['sd_wealth_real_wave10_male'],
						moments_female['sd_wealth_real_wave10_female'],
						moments_male['sd_wealth_fin_wave14_male'],
						moments_female['sd_wealth_fin_wave14_female'],
						moments_male['sd_wealth_fin_wave10_male'],
						moments_female['sd_wealth_fin_wave10_female'],
						moments_male['mean_vol_cont_c_wave14_male'],
						moments_female['mean_vol_cont_c_wave14_female'],
						moments_male['mean_vol_cont_c_wave10_male'],
						moments_female['mean_vol_cont_c_wave10_female'],
						moments_male['mean_risky_share_wave14_male'],
						moments_female['mean_risky_share_wave14_female'],
						moments_male['mean_risky_share_wave10_male'],
						moments_female['mean_risky_share_wave10_female'],
						moments_male['mean_vol_cont_adj_wave14_male'],
						moments_female['mean_vol_cont_adj_wave14_female'],
						moments_male['mean_vol_cont_adj_wave10_male'],
						moments_female['mean_vol_cont_adj_wave10_female'],
						moments_male['mean_vol_total_wave14_male'],
						moments_female['mean_vol_total_wave14_female'],
						moments_male['mean_vol_total_wave10_male'],
						moments_female['mean_vol_total_wave10_female'],
						moments_male['mean_wealth_real_wave14_male'],
						moments_female['mean_wealth_real_wave14_female'],
						moments_male['mean_wealth_real_wave10_male'],
						moments_female['mean_wealth_real_wave10_female'],
						moments_male['mean_wealth_fin_wave14_male'],
						moments_female['mean_wealth_fin_wave14_female'],
						moments_male['mean_wealth_fin_wave10_male'],
						moments_female['mean_wealth_fin_wave10_female'],
						moments_male['mean_super_balance_wave14_male'],
						moments_female['mean_super_balance_wave14_female'],
						moments_male['mean_super_balance_wave10_male'],
						moments_female['mean_super_balance_wave10_female'],
						moments_male['mean_consumption_wave14_male'],
						moments_female['mean_consumption_wave14_female'],
						moments_male['mean_consumption_wave10_male'], 
						moments_female['mean_consumption_wave10_female']], axis =1)

 	return moments_sorted 

#Generate plots

if __name__ == '__main__':



	simlist_f = ['female_4']
	simlist_m = ['male_16']

	sim_id = "male_16"

	for sim_id_m, sim_id_f  in zip(simlist_m,simlist_f) :
		if not os.path.isdir(sim_id):
			os.makedirs(sim_id)

		plt.close()

		sns.set(font_scale =1.5,style='ticks',rc={"lines.linewidth": 0.7, \
			"axes.grid.axis":"both","axes.linewidth":2,"axes.labelsize":18})
		plt.rcParams["figure.figsize"] = (20,10)
		sns.set_style('white')
		
		

		#og_male = pickle.load(open("/scratch/pv33/baseline_male.mod","rb")) 
		#og_female = pickle.load(open("/scratch/pv33/baseline_female.mod","rb")) 
		
		#og_male.accountdict 		= {}
		#og_male.accountdict[1] 		= 'DC'
		#og_male.accountdict[0] 		= 'DB'
		#og_female.accountdict 		= {}
		#og_female.accountdict[1] 		= 'DC'
		#og_female.accountdict[0] 		= 'DB'


		def create_plot(df,col1,col2, source, marker,color, ylim, ylabel):
		    #df[col1] = df[col1].map(lambda x :inc_y0(x))
		    #sns.set()
		     #list(set(df[source]))
		    line_names =['wave14_data', 'wave10_data', 'wave14_sim', 'wave10_sim']
		    linestyles=["-","-","-","-"]
		    col_dict = {'wave14_data': 'black', 'wave10_data':'black', 'wave14_sim':'gray', 'wave10_sim':'gray'}

		    normalise_list = ['sd_vol_totalDC','sd_vol_totalDB','sd_super_balanceDC','sd_super_balanceDB','mean_wealth_real','mean_wealth_fin',\
					'mean_super_balance',\
					'mean_vol_total',\
					'mean_vol_cont_c',\
	                'mean_consumption','sd_consumption', \
	                'sd_wealth_real', 'sd_wealth_fin']


		    df_male = df.iloc[:,[0, 1,3,5,7]]
		    df_female = df.iloc[:,[0, 2,4,6,8]]	


		    df_male.columns = df_male.columns.str.replace("_male", "")
		    df_female.columns = df_female.columns.str.replace("_female", "")



		    df_male =df_male.melt('Age_wave10', var_name= 'source', value_name = key)
		    df_male['source'] = df_male['source'].str.replace(key+"_", "")
		    df_female =df_female.melt('Age_wave10', var_name= 'source', value_name = key)
		    df_female['source'] = df_female['source'].str.replace(key+"_", "")
		    

		    markers=['x', 'o', 'x', 'o']

		    if col2 in normalise_list:
		    	df_male[col2] = df_male[col2].div(1000)
		    	df_female[col2] = df_female[col2].div(1000)

		    figure, axes = plt.subplots(nrows=1, ncols=2)

		    for name, marker, linestyle in zip(line_names, markers, linestyles):
	                    data_male = df_male.loc[df_male[source]==name]
	                    data_female = df_female.loc[df_female[source]==name]
	                    xs = list(data_male[col1])[0:18]
	                    ys = list(data_male[col2])
	                    p = axes[0].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
	                                 label=name, linewidth=2)
	                    ys = list(data_female[col2])
	                    p = axes[1].plot(xs, ys, marker=marker, color=col_dict[name], linestyle=linestyle,
	                                label=name, linewidth=2)

	        
		    if isinstance(ylabel, str):
	                    ylabel = ylabel
		    
		    axes[0].set_title('Males')
		    axes[0].set_xlabel('Age cohort')
		    #axes[0].set_ylabel(ylabel)
		    axes[0].set_ylim(ylim)
		    axes[0].spines['top'].set_visible(False)
		    axes[0].spines['right'].set_visible(False)
		    axes[0].legend(loc='upper left', ncol=2)
		    #plt.show()
		    axes[1].set_title('Females')
		    axes[1].set_xlabel('Age cohort')
		   #axes[1].set_ylabel(ylabel)
		    axes[1].set_ylim(ylim)
		    axes[1].spines['top'].set_visible(False)
		    axes[1].spines['right'].set_visible(False)
		    axes[1].legend(loc='upper left', ncol=2)
		    #plt.tight_layout()
		    #figure.size(10,10)
		    figure.savefig("{}/{}.png".format(sim_id,col2), transparent=True)
		   

		#TSALL_10_male,TSALL_14_male  = genprofiles(og_male, N = 100)

		#pickle.dump(TSALL_10_male, open("/scratch/pv33/TSALL_10_male.mod","wb"))
		#pickle.dump(TSALL_14_male, open("/scratch/pv33/TSALL_14_male.mod","wb"))

		#TSALL_10_female,TSALL_14_female  =genprofiles(og_female, N = 100)


		#pickle.dump(TSALL_10_female, open("/scratch/pv33/TSALL_10_female.mod","wb"))
		#pickle.dump(TSALL_14_female, open("/scratch/pv33/TSALL_14_female.mod","wb"))


		TSALL_10_male, TSALL_14_male = pickle.load(open("TSALL_10_{}.mod".format(sim_id_m),"rb")), pickle.load(open("TSALL_14_{}.mod".format(sim_id_m),"rb"))
		TSALL_10_female, TSALL_14_female = pickle.load(open("TSALL_10_{}.mod".format(sim_id_f),"rb")), pickle.load(open("TSALL_14_{}.mod".format(sim_id_f),"rb")) 

		 
		moments_male = gen_moments(TSALL_10_male,TSALL_14_male)
		#moments_male = pd.read_csv('moments_male_old.csv')
		moments_female = gen_moments(TSALL_10_female,TSALL_14_female)

		moments_female = moments_female.add_suffix('_female')
		moments_male = moments_male.add_suffix('_male')
		
		moments_male.to_csv("/scratch/pv33/moments_male.csv") 
		moments_female.to_csv("/scratch/pv33/moments_female.csv") 

		moments_sorted 	= sortmoments(moments_male, moments_female)
		

		moments_sorted.to_csv("{}/moments_sorted.csv".format(sim_id))  

		moments_sorted = pd.concat([moments_male["Age_wave10_male"].reset_index().iloc[:,1], moments_sorted], axis =1)  

		moments_sorted = moments_sorted.rename(columns = {'Age_wave10_male':'Age_wave10'})
		
		moments_data = pd.read_csv('moments_data.csv')
		moments_data = moments_data.drop('Unnamed: 0', axis=1)   

		moments_data.columns = moments_sorted.columns



		age 		= np.arange(18, 65) # PROBABLY SHOULD GET RID OF AGE MAGIC NUMBERS HERE 

		plot_keys_vars 	= [	'mean_account_type',
							'corr_super_balance_risky_share_adj_DC',
							'corr_super_balance_risky_share_adj_DB',
							'corr_super_balance_risky_share_DC',
							'corr_super_balance_risky_share_DB',
							'corr_super_balance_vol_cont_adj_DC',
							'corr_super_balance_vol_cont_adj_DB',
							'corr_vol_total_super_balance_DC',
							'corr_vol_total_super_balance_DB',
							'corr_consumption_wealth_real',
							'sd_risky_shareDC',
							'sd_risky_shareDB',
							'sd_vol_totalDC',
							'sd_vol_totalDB',
							'sd_super_balanceDC',
							'sd_super_balanceDB',
							'sd_wealth_real',
							'sd_wealth_fin',
							'mean_risky_share',
							'mean_vol_cont_adj',
							'mean_vol_total',
							'mean_vol_cont_c',
							'mean_wealth_real',
							'mean_wealth_fin',
							'mean_super_balance',
							'mean_consumption', 
							'sd_consumption']




		plot_autocors = ['wealth_real_autocorr',
							'wealth_fin_autocorr',
							'consumption_autocorr']
		

		# variables with y axis 0:1000
		# real wealth, fin wealth super balance
		axis_style_list_ylim = {
							'mean_account_type':(0,1),
							'sd_super_balanceDC':(0,1500),
							'sd_super_balanceDB':(0,1500),
							'sd_wealth_real':(0,1500),
							'sd_wealth_fin':(0,1500),
							'mean_wealth_real':(0,1500),
							'mean_wealth_fin':(0,1500),
							'mean_super_balance':(0,1500),
		# varibales with y 1- 150 (consumption)
							'mean_consumption':(0,150),
							'sd_consumption':(0,150),
		# varibales with y axis 0 -1(risky share and vol cont)
							'mean_vol_cont_adj':(0,1),
							'mean_risky_share':(0,1),
							'sd_risky_shareDC':(0,1),
							'sd_risky_shareDB':(0,1),
							'sd_vol_totalDC':(0,15),
							'sd_vol_totalDB':(0,15),
							'mean_vol_total': (0,15),
							'mean_vol_cont_c': (0,15),
		# varibales with y axis -.5 to 1(correlations)
							'corr_super_balance_risky_share_adj_DC':(-.8,1),
							'corr_super_balance_risky_share_adj_DB':(-.8,1),
							'corr_super_balance_risky_share_DC':(-.8,1),
							'corr_super_balance_risky_share_DB':(-.8,1),
							'corr_super_balance_vol_cont_adj_DC':(-.8,1),
							'corr_super_balance_vol_cont_adj_DB':(-.8,1),
							'corr_vol_total_super_balance_DC':(-.8,1),
							'corr_vol_total_super_balance_DB':(-.8,1),
							'corr_consumption_wealth_real':(-.8,1)}

		axis_label_list = {
							'mean_account_type':'Share of DC',
							'sd_super_balanceDC':'SD: UniSuper balance (DC)',
							'sd_super_balanceDB':'SD: UniSuper balance (DB)',
							'mean_wealth_real': 'Housing asset',
							'mean_wealth_fin':'Asset',
							'mean_super_balance':'UniSuper balance (DB+DC)',
							'sd_wealth_real':'SD: Asset',
							'sd_wealth_fin' :'SD: Housing asset',
		# varibales with y 1- 150 (consumption)
							'mean_consumption':'Consumption',
							'sd_consumption': 'SD: Consumption',
		# varibales with y axis 0 -1(risky share and vol cont)
							'mean_vol_cont_adj':'Share of positive voluntary contribution',
							'mean_risky_share': 'Share of Unisuper balance in risky assets',
							'sd_risky_shareDC':	'SD: Share of Unisuper balance in risky assets among DCer',
							'sd_risky_shareDB':	'SD: Share of Unisuper balance in risky assets among DBer',
							'sd_vol_totalDC':'SD: total voluntary contribution among DCer',
							'sd_vol_totalDB':'SD: total voluntary contribution among DBer',
							'mean_vol_total': 'Total voluntary contribution',
							'mean_vol_cont_c':'Total voluntary contribution (among contributors)',
		# varibales with y axis -.5 to 1(correlations)
							'corr_super_balance_risky_share_adj_DC':'CORR: UniSuper balane and non−default inv among DCer',
							'corr_super_balance_risky_share_adj_DB':'CORR: UniSuper balane and non−default inv among DBer',
							'corr_super_balance_risky_share_DC':'CORR: UniSuper balane and risky share among DCer',
							'corr_super_balance_risky_share_DB':'CORR: UniSuper balane and risky share among DCer',
							'corr_super_balance_vol_cont_adj_DC':'CORR: UniSuper balane and +vc among DCer',
							'corr_super_balance_vol_cont_adj_DB':'CORR: UniSuper balane and +vc among DBer',
							'corr_vol_total_super_balance_DC':'CORR: UniSuper balane and vc among DCer',
							'corr_vol_total_super_balance_DB':'CORR: UniSuper balane and vc among DCer',
							'corr_consumption_wealth_real':'CORR: Consumption and real wealth'}


		#excludes 	= set(['account_type_all'])

		cohort_labels = pd.DataFrame(list(["<25",
						"25-29",
						"30-34",
						"35-39",
						"40-44",
						"45-49",
						"50-54",
						"55-59",
						"60+",
						]))
		cohort_labels.columns= ['Age_wave10'] 

		for  key in plot_keys_vars:

			#Assets 
			var_data = moments_data[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
			var_data = var_data.add_suffix('_data')
			var_data.iloc[8,2] = float('nan')
			var_data.iloc[8,3] =float('nan')  


			var_sim = moments_sorted[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
			var_sim = var_sim.add_suffix('_sim')
			var_sim.iloc[8,2] = float('nan')
			var_sim.iloc[8,3] =float('nan')  


			var_grouped = pd.concat([cohort_labels, var_data, var_sim], axis =1)
			
	 
			ylim = axis_style_list_ylim[key]

			create_plot(var_grouped, 'Age_wave10', key, 'source', marker='s',color='darkblue', ylim = ylim, ylabel = axis_label_list[key])


