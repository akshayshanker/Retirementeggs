
"""
This module contains the functions to generate time series of agents
and moments from a solved HousingModel 
 
Functions: genprofiles
            Generates the a time series of N agents from the HousingModel class 
           
           housingmodel_function_factory
            Generates moments from time-series 

"""
"i made this edit in testbranch2_only"
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

from profiles_moments import genprofiles, gen_moments, sortmoments



for cf in ['cf1', 'cf2', 'cf3','cf4', 'cf5', 'cf6']:


	simlist_f = ['female_4','female_'+cf]
	simlist_m = ['male_16', 'male_'+cf]
	cf_list   = ['baseline', cf]



	cf_name = 'counterfactual'
	sim_id = cf_name


	#define plot function 
	figure, axes = plt.subplots(2, 3)


	def create_plot(df,col1,col2, source, marker,color, ylim, ylabel,i):
	    #df[col1] = df[col1].map(lambda x :inc_y0(x))
	    #sns.set()
	     #list(set(df[source]))
	    
	    col_dict = {'wave10_sim': 'gray', 'wave14_sim': 'gray', 'wave10_'+cf_name: 'black', 'wave14_'+cf_name: 'black'}
	    line_names =['wave10_sim', 'wave14_sim', 'wave10_'+cf_name, 'wave14_'+cf_name]


	    df_male = df.iloc[:,[0, 1,3,5,7,9,11]]
	    df_female = df.iloc[:,[0, 2,4,6,8,10,12]]	


	    df_male.columns = df_male.columns.str.replace("_male", "")
	    df_female.columns = df_female.columns.str.replace("_female", "")



	    df_male =df_male.melt('Age_wave10', var_name= 'source', value_name = key)
	    df_male['source'] = df_male['source'].str.replace(key+"_", "")
	    df_female =df_female.melt('Age_wave10', var_name= 'source', value_name = key)
	    df_female['source'] = df_female['source'].str.replace(key+"_", "")
	    

	    markers=['o', 'x', 'o', 'x']

	    #if col2 in normalise_list
	    df_male[col2] = df_male[col2].div(1000)
	    df_female[col2] = df_female[col2].div(1000)

	    for name, marker in zip(line_names, markers):
	                data_male = df_male.loc[df_male[source]==name]
	                data_female = df_female.loc[df_female[source]==name]
	                xs = list(data_male[col1])[0:18]
	                ys = list(data_male[col2])
	                p = axes[0,i].plot(xs, ys, marker=marker, color=col_dict[name], linestyle="-",
	                             label=name, linewidth=2)
	                ys = list(data_female[col2])
	                p = axes[1,i].plot(xs, ys, marker=marker, color=col_dict[name], linestyle="-",
	                            label=name, linewidth=2)

	    
	    if isinstance(ylabel, str):
	                ylabel = ylabel
	    
	    axes[0,i].set_title(axis_label_list[key])
	    #axes[0,i].set_xlabel('Age cohort')
	    #axes[0].set_ylabel(ylabel)
	    axes[0,i].set_ylim(ylim)
	    axes[0,i].spines['top'].set_visible(False)
	    axes[0,i].spines['right'].set_visible(False)
	    axes[0,i].tick_params(axis='x', rotation=45)
	    axes[0,i].set_xlabel('Age cohort')
	    #axes[0,i].legend(loc='upper left', ncol=2)
	    #plt.show()
	    #axes[1,i].set_title('Females')
	    axes[1,i].set_xlabel('Age cohort')
	    #axes[1,i].set_title(axis_label_list[key])
	   #axes[1].set_ylabel(ylabel)
	    axes[1,i].set_ylim(ylim)
	    axes[1,i].spines['top'].set_visible(False)
	    axes[1,i].spines['right'].set_visible(False)
	    #axes[1,i].legend(loc='upper left', ncol=2)
	    #plt.tight_layout()
	    #figure.size(10,10)
	    axes[1,i].tick_params(axis='x', rotation=45) 
	    


	#create dictionary of moments 

	moments_dict ={}
	moments_data = pd.read_csv('moments_data.csv')
	moments_data = moments_data.drop('Unnamed: 0', axis=1)  
	moments_dict['data'] = moments_data




	for sim_id_m, sim_id_f, cf_id  in zip(simlist_m,simlist_f, cf_list) :
		if not os.path.isdir(sim_id):
			os.makedirs(sim_id)

		plt.close()

		sns.set(font_scale =1.5,style='ticks',rc={"lines.linewidth": 0.7, \
			"axes.grid.axis":"both","axes.linewidth":2,"axes.labelsize":18})
		plt.rcParams["figure.figsize"] = (20,10)
		sns.set_style('white')
		


		TSALL_10_male, TSALL_14_male = pickle.load(open("TSALL_10_{}.mod".format(sim_id_m),"rb")), pickle.load(open("TSALL_14_{}.mod".format(sim_id_m),"rb"))
		TSALL_10_female, TSALL_14_female = pickle.load(open("TSALL_10_{}.mod".format(sim_id_f),"rb")), pickle.load(open("TSALL_14_{}.mod".format(sim_id_f),"rb")) 

		 
		moments_male = gen_moments(TSALL_10_male,TSALL_14_male)
		#moments_male = pd.read_csv('moments_male_old.csv')
		moments_female = gen_moments(TSALL_10_female,TSALL_14_female)

		moments_female = moments_female.add_suffix('_female')
		moments_male = moments_male.add_suffix('_male')
		
		moments_male.to_csv("{}/moments_{}.csv".format(sim_id, sim_id_m)) 
		moments_female.to_csv("{}/moments_{}.csv".format(sim_id, sim_id_f)) 

		moments_sorted 	= sortmoments(moments_male, moments_female)
		

		moments_sorted.to_csv("{}/moments_sorted_{}.csv".format(sim_id,sim_id_m.replace('male', '')))  

		moments_sorted = pd.concat([moments_male["Age_wave10_male"].reset_index().iloc[:,1], moments_sorted], axis =1)  

		moments_sorted = moments_sorted.rename(columns = {'Age_wave10_male':'Age_wave10'})
		
	 

		moments_data.columns = moments_sorted.columns

		moments_dict[cf_id] = moments_sorted 



	#now make plots 
	age 		= np.arange(18, 65) # PROBABLY SHOULD GET RID OF AGE MAGIC NUMBERS HERE 

	plot_keys_vars 	= [
						'mean_super_balance',
						'mean_wealth_fin',
						'mean_wealth_real']

	plot_tiles = {'mean_super_balance':'Pension wealth','mean_super_balance':'Financial wealth','mean_super_balance':'Housing wealth'}




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
						'mean_wealth_real': 'Housing wealth',
						'mean_wealth_fin':'Financial wealth',
						'mean_super_balance':'Pension wealth',
						'sd_wealth_real':'SD: Financial wealth',
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

	i = 0

	plt.close()
	#define plot function 
	figure, axes = plt.subplots(2, 3)

	for  key in plot_keys_vars:

		#Assets 
		var_data = moments_data[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
		var_data = var_data.add_suffix('_data')
		#var_data.iloc[8,2] = float('nan')
		#var_data.iloc[8,3] =float('nan')  
		var_grouped = pd.concat([cohort_labels, var_data], axis =1)


		for cf_id in cf_list:
			moments_sorted = moments_dict[cf_id]
			var_sim = moments_sorted[[key+'_wave10_male',key+'_wave10_female',key+'_wave14_male',key+'_wave14_female']]
			var_sim = var_sim.add_suffix('_sim')
			var_sim.iloc[8,2] = float('nan')
			var_sim.iloc[8,3] =float('nan')  
			var_grouped = pd.concat([var_grouped, var_sim], axis =1)

		var_grouped.columns= list(['Age_wave10', 'data', 'data', 'data', 'data', 
									'wave10_sim', 'wave10_sim','wave14_sim', 'wave14_sim',
									'wave10_'+cf_name, 'wave10_'+cf_name,'wave14_'+cf_name, 'wave14_'+cf_name])



		ylim = axis_style_list_ylim[key]

		create_plot(var_grouped, 'Age_wave10', key, 'source', marker='s',color='darkblue', ylim = ylim, ylabel = axis_label_list[key], i = i)
		i = i+1

	handles, labels = axes[1,2].get_legend_handles_labels()
	#figure.legend(handles, labels, loc='upper center')




	#for ax in axes.flat:
	#    ax.label_outer()

	axes[0,0].set_ylabel('Males')
	axes[1,0].set_ylabel('Females')


	figure.subplots_adjust( hspace=.7, bottom=0.2, wspace=0.33) 
	plt.legend(frameon=False)

	axes[1,1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=4)

	figure.savefig("{}_plots.png".format(cf), transparent=True)

