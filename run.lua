require 'nn'
require 'cudnn'
require 'cunn'
require 'hdf5'
require 'torch'
require 'optim'
require 'nngraph'

collectgarbage()
--Data is loaded using hdf5 format. The paths to the data are
--specified in the opts file.


--load the command line options
--load the model and the loss criterion
local opts = require 'opts'



-- Get the input arguments parsed and stored in opt
opt = opts.parse(arg)


torch.setdefaulttensortype('torch.FloatTensor')
-- if opt.gpu==1 then
-- 	torch.setdefaulttensortype('torch.CudaTensor')
-- else
-- 	torch.setdefaulttensortype('torch.FloatTensor')
-- end



if opt.runmode==1 then
	require 'train'
	
	train_model(opt)
	
	

else

	require 'test'
	test_model(opt)


end


collectgarbage()

