local opts={}


--call the parse function to get the cmd line options. 
--The default values are set in this file. 
--To change them pass these value with corresponding flag in cmd line.
--eg th opts.lua -learning_rate 0.01
function opts.parse(arg)
	cmd=torch.CmdLine()
	cmd:text()
	cmd:text('Train a segmentation model.')
	cmd:text('Options')

	--Backend settings
	cmd:option('-backend','cudnn','nn|cudnn')
	cmd:option('-threads',8,'set the number of threads')
	--test or train
	cmd:option('-runmode',1,'train if mode=1 and test if mode=0')
	cmd:option('-gpu',1,'use gpu else use cpu')

	--filepaths

	cmd:option('-hdf5_train_path','./dataset.hdf5','path to the hdf5 for training')
	cmd:option('-model_path','./mytrain.th','path to the trained model')
	cmd:option('-hdf5_test_path','./test/testset.hdf5','path of the test image directory')
	--Hyperparameters
	cmd:option('-weightdecay',1e-5,'L2 weight decay penalty strength')
	cmd:option('-batch_size',10,'minibatch size')
	cmd:option('-learning_rate',1e-4,'learning_rate')
	cmd:option('-optim_beta1',0.9,'beta1 for adam')
	cmd:option('-optim_beta2',0.999,'beta2 for adam')
	cmd:option('-optim_epsilon', 1e-8, 'epsilon for smoothing')
  	cmd:option('-drop_prob', 0.5, 'Dropout strength throughout the model.')
  	cmd:option('-max_iters', 10000, 'Number of iterations to run; -1 to run forever')
  	cmd:option('-checkpoint_start_from', '','Load model from a checkpoint instead of random initialization.')
  	cmd:option('-finetune_cnn_after', -1,'Start finetuning CNN after this many iterations (-1 = never finetune)')
  	cmd:option('-save_checkpoint_every', 10000,'How often to save model checkpoints')
  	cmd:option('-losses_log_every', 10,'How often do we save losses, for inclusion in the progress dump? (0 = disable)')
	cmd:text()
  	local opt = cmd:parse(arg or {})

  	return opt
end	

return opts
