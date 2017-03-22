require 'hdf5'
require 'nn'
require 'nngraph'
require 'cudnn'
require 'optim'
require 'torch'
require 'cutorch'
require 'cunn'
require 'gnuplot'
require 'xlua'
local nninit=require 'nninit'
torch.setdefaulttensortype('torch.FloatTensor')

function train_model(opt)

---------------------MODEL DEFINITION------------------
collectgarbage()
torch.setnumthreads(opt.threads)

--[[Model declaration in the model.lua and importing using function call gives an error /torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
	[C]: at 0x00406670
--]]	
SegModel=nn.Sequential()
--module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH])
SegModel:add(cudnn.SpatialConvolution(3,3,9,1,3,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(3, 1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(3,6,1,9,1,3):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(6,1e-3))
SegModel:add(cudnn.ReLU(true))

SegModel:add(cudnn.SpatialConvolution(6,15,1,5,1,2):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(15,1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(15,3,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(3,1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(3,15,5,1,2,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(5,1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(15,3,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(3,1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(3,5,3,3,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(5,1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(5,1,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
SegModel:add(nn.SpatialBatchNormalization(1,1e-3))
SegModel:add(cudnn.ReLU(true))


--nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH], [adjW], [adjH])
SegModel:add(cudnn.SpatialFullConvolution(1,16,3,3,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}):noBias())
SegModel:add(nn.SpatialBatchNormalization(16, 1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialFullConvolution(16,32,5,5,2,2):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}):noBias())
SegModel:add(nn.SpatialBatchNormalization(32, 1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialFullConvolution(32,32,12,12,3,3):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}):noBias())
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(32,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))


--SegModel:add(nn.SpatialBatchNormalization(32,1e-3))
--SegModel:add(cudnn.ReLU(true))
--SegModel:add(cudnn.SpatialFullConvolution(32,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}):noBias())
--SegModel=nn.gModule({})
criterion=cudnn.SpatialCrossEntropyCriterion()

SegModel=SegModel:cuda()
criterion:cuda()


--[[
    This criterion does the SpatialCrossEntropyCriterion across
    the feature dimension for a N-channel image of HxW in size.
    It only supports mini-batches (4D input, 3D target)
    It does a LogSoftMax on the input (over the channel dimension),
    so no LogSoftMax is needed in the network at the end
    input = batchSize x nClasses x H x W
    target = batchSize x H x W
]]--

-------------------------OPEN HDF5 FILE-------------------
local myFile=hdf5.open(opt.hdf5_train_path)
local data = myFile:read('/dataset'):all()
--close the hdf5 file
myFile:close()

print(SegModel)

-----------------------------------------------------------
-- To convert hdf5 data to table with numbered keys
--[[ 
train={}
label={}
for key,value in pairs(data['train']) do
	train[tonumber(key)]=value
	label[tonumber(key)]=data['label'][key]
	data['label'][key]=nil
	data['train'][key]=nil
end
data=nil
--]]


-- Convert hdf5 data to tensor.
train=torch.Tensor()
label=torch.Tensor()
for key,value in pairs(data['train']) do
	train=torch.cat(train,value:float(),4)
	label=torch.cat(label,data['label'][key]:float(),3)
	data['label'][key]=nil
	data['train'][key]=nil
end
data=nil
train=train:permute(4,3,1,2):cuda()
label=label:permute(3,1,2):cuda()


logger=optim.Logger('loss.log')
logger:setNames{'Spatial Cross Entropy Loss'}
logger:style{'+-'}

params,grad_params = SegModel:getParameters()


--set the hyperparameters for ADAM
--check optim.adam for code


config={learningRate=opt.learning_rate,beta1=opt.optim_beta1,beta2=opt.optim_beta2,
epsilon=opt.optim_epsilon}
SegModel:zeroGradParameters()

---------------------TRAINING LOOP------------------
for epoch=1,opt.max_iters do
	xlua.progress(epoch, opt.max_iters)
	for batch=1,(100-opt.batch_size),opt.batch_size do
		xlua.progress(batch, 100)

		cutrain=train[{{batch,batch+opt.batch_size},{},{},{}}]:cuda()
		culabel=label[{{batch,batch+opt.batch_size},{},{}}]:cuda()
		local feval = function(x)
			if x ~= parameters then params:copy(x) end
			grad_params:zero()
			pred=SegModel:forward(cutrain)
			err=criterion:forward(pred,culabel) 
			df_do=criterion:backward(pred,culabel)
			df_di=SegModel:backward(cutrain,df_do)

			return err,grad_params
		end

		

		--SegModel:updateParameters(opt.learning_rate)
		
		optim.adam(feval, params, config)
		
		logger:add{err}
		logger:plot()
		collectgarbage()
	end
	print("Finished epoch",epoch)
	if epoch%50==0 then
		torch.save('mytrain.th', SegModel)
		gnuplot.pngfigure('plot.png')
	end

collectgarbage()
end




--Sample code for plotting the error and accuracy logs
--[[
logger=optim.Logger('accuracy.log')
logger:setNames{'Training acc.', 'Test acc.'}
for i = 1, 10 do
   trainAcc = math.random(0, 100)
   testAcc = math.random(0, 100)
   logger:add{trainAcc, testAcc}
end
logger:style{'+-', '+-'}
logger:plot()  --does not accept CudaTensor, either change the deafault tensor and send everything to GPU and back or find a way around
logger:display(false)
--]]




--torch.save('mytrain'..tostring(iter)..'_'..tostring(err)..'.th', model)
torch.save('mytrain.th', SegModel)
collectgarbage()
end





