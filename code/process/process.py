from omegaconf import OmegaConf 
import argparse
import os

from data_splitting import main as split

if __name__=='__main__':

	#---------- Read config file ----------#
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_file", type=str, help="Path for the splitting config file", required=True) 
	args_m = parser.parse_args()

	## Read config file
	if not(os.path.exists(args_m.config_file)):         
	    raise RuntimeError("config_file {} does not exist".format(args_m.config_file))

	# split the data with given config
	config = OmegaConf.load(args_m.config_file)


	if not(os.path.exists(config.config_files.segmenting)):         
	    raise RuntimeError("The config file of the segmenting step '{}' does not exist"
	    	.format(config.config_files.segmenting))

	if not(os.path.exists(config.config_files.cleaning)):         
	    raise RuntimeError("The config file of the cleaning step '{}' does not exist"
	    	.format(config.config_files.cleaning))

	flag_split = 0
	if not(os.path.exists(config.config_files.splitting)) or not(os.path.exists(config.config_files.splitting_feats)):
		if os.path.exists(config.config_files.splitting):
			print("Only signal data will be splitted.") 
			print("The config file of feature-based data for splitting '{}' does not exist or was not provided."
				.format(config.config_files.splitting_feats))
			flag_split = 1
		elif os.path.exists(config.config_files.splitting_feats):
			print("Only feature-based data will be splitted.") 
			print("The config file of signal data for splitting '{}' does not exist or was not provided."
				.format(config.config_files.splitting))
			flag_split = 2
		else:
			raise RuntimeError("the config files for data splitting '{}' & '{}' does not exist. "
				.format(config.config_files.splitting, config.config_files.splitting_feats))


	#---------- Read, alignment and segmentation of raw data ----------#
	print('Read, alignment and segmentation of raw data...\n')

	if config.data_name == 'BCG':
		from read_bcg import main as read
	elif config.data_name == 'PPGBP':
		from read_ppgbp import main as read
	elif config.data_name == 'sensors':
		from read_sensors import main as read
	elif config.data_name == 'UCI':
		from read_uci import main as read
	else:
		raise RuntimeError("Incorrect data name {}. It should be BCG, PPGBP, sensors or UCI."
				.format(config.data_name))

	config_read = OmegaConf.load(config.config_files.segmenting)
	read(config_read)

	print('\n\n')


	#---------- Cleaning and feature generation ----------#
	print('Cleaning and feature generation...\n')

	if config.data_name == 'PPGBP':
		from cleaningPPGBP import main as clean
	else:
		from cleaning import main as clean

	config_clean = OmegaConf.load(config.config_files.cleaning)
	clean(config_clean)

	print('\n\n')


	#---------- Data splitting for validation ----------#
	print('Data splitting for validation...\n')

	if flag_split != 1:
		print(' - Splitting feature-based data...\n')
		config_split = OmegaConf.load(config.config_files.splitting_feats)
		split(config_split)
	if flag_split != 2:
		print(' - Splitting signal-based data...\n')
		config_split = OmegaConf.load(config.config_files.splitting)
		split(config_split)

	print('\n\n')




