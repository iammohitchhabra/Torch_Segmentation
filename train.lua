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
require 'nn'
require 'graph'

local nninit=require 'nninit'

function train_model(opt)

---------------------MODEL DEFINITION------------------
collectgarbage()
torch.setnumthreads(opt.threads)

--[[Model declaration in the model.lua and importing using function call gives an error /torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
	[C]: at 0x00406670
--]]	

SegModel=nn.Sequential()
--module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH])

----------------------------ENCODER-------------------------------------------
initial=nn.ConcatTable()
i1=nn.Sequential()
i1:add(cudnn.SpatialConvolution(3,13,3,3,2,2,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
i2=nn.Sequential()
i2:add(cudnn.SpatialMaxPooling(3,3,2,2,1,1))
initial:add(i1)
initial:add(i2)
SegModel:add(initial)
SegModel:add(nn.JoinTable(2,4))

bottleneck1=nn.ConcatTable()
bt1_1=nn.Sequential()
bt1_1:add(cudnn.SpatialConvolution(16,16,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt1_1:add(nn.SpatialBatchNormalization(16, 1e-3))
bt1_1:add(cudnn.ReLU(true))
bt1_1:add(cudnn.SpatialConvolution(16,32,3,3,2,2,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt1_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt1_1:add(cudnn.ReLU(true))
bt1_1:add(cudnn.SpatialConvolution(32,16,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt1_1:add(nn.SpatialBatchNormalization(16, 1e-3))
bt1_1:add(cudnn.ReLU(true))
bt1_1:add(nn.SpatialDropout(0.1))
bt1_2=nn.Sequential()
bt1_2:add(cudnn.SpatialMaxPooling(3,3,2,2,1,1))
bottleneck1:add(bt1_1)
bottleneck1:add(bt1_2)
SegModel:add(bottleneck1)
SegModel:add(nn.JoinTable(2,4))

bottleneck2=nn.ConcatTable()
bt2_1=nn.Sequential()
bt2_1:add(cudnn.SpatialConvolution(32,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt2_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt2_1:add(cudnn.ReLU(true))
bt2_1:add(cudnn.SpatialConvolution(32,32,3,3,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt2_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt2_1:add(cudnn.ReLU(true))
bt2_1:add(cudnn.SpatialConvolution(32,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt2_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt2_1:add(cudnn.ReLU(true))
bt2_1:add(nn.SpatialDropout(0.1))
bt2_2=nn.Sequential()
bt2_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck2:add(bt2_1)
bottleneck2:add(bt2_2)
SegModel:add(bottleneck2)
SegModel:add(nn.JoinTable(2,4))

bottleneck3=nn.ConcatTable()
bt3_1=nn.Sequential()
bt3_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt3_1:add(nn.SpatialBatchNormalization(64, 1e-3))
bt3_1:add(cudnn.ReLU(true))
bt3_1:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt3_1:add(nn.SpatialBatchNormalization(64,1e-3))
bt3_1:add(cudnn.ReLU(true))
bt3_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt3_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt3_1:add(cudnn.ReLU(true))
bt3_1:add(nn.SpatialDropout(0.1))
bt3_2=nn.Sequential()
bt3_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt3_2:add(nn.SpatialBatchNormalization(32, 1e-3))
bt3_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck3:add(bt3_1)
bottleneck3:add(bt3_2)
SegModel:add(bottleneck3)
SegModel:add(nn.JoinTable(2,4))

bottleneck4=nn.ConcatTable()
bt4_1=nn.Sequential()
bt4_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt4_1:add(nn.SpatialBatchNormalization(64, 1e-3))
bt4_1:add(cudnn.ReLU(true))
bt4_1:add(nn.SpatialDilatedConvolution(64,64,3,3,1,1,2,2,2,2))
bt4_1:add(nn.SpatialBatchNormalization(64,1e-3))
bt4_1:add(cudnn.ReLU(true))
bt4_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt4_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt4_1:add(cudnn.ReLU(true))
bt4_1:add(nn.SpatialDropout(0.1))
bt4_2=nn.Sequential()
bt4_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt4_2:add(nn.SpatialBatchNormalization(32, 1e-3))
bt4_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck4:add(bt4_1)
bottleneck4:add(bt4_2)
SegModel:add(bottleneck4)
SegModel:add(nn.JoinTable(2,4))

bottleneck5=nn.ConcatTable()
bt5_1=nn.Sequential()
bt5_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt5_1:add(nn.SpatialBatchNormalization(64, 1e-3))
bt5_1:add(cudnn.ReLU(true))
bt5_1:add(cudnn.SpatialConvolution(64,64,5,1,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt5_1:add(cudnn.SpatialConvolution(64,64,1,5,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt5_1:add(nn.SpatialBatchNormalization(64,1e-3))
bt5_1:add(cudnn.ReLU(true))
bt5_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt5_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt5_1:add(cudnn.ReLU(true))
bt5_1:add(nn.SpatialDropout(0.1))
bt5_2=nn.Sequential()
bt5_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt5_2:add(nn.SpatialBatchNormalization(32, 1e-3))
bt5_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck5:add(bt5_1)
bottleneck5:add(bt5_2)
SegModel:add(bottleneck5)
SegModel:add(nn.JoinTable(2,4))

bottleneck6=nn.ConcatTable()
bt6_1=nn.Sequential()
bt6_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt6_1:add(nn.SpatialBatchNormalization(64, 1e-3))
bt6_1:add(cudnn.ReLU(true))
bt6_1:add(nn.SpatialDilatedConvolution(64,64,3,3,1,1,4,4,4,4))
bt6_1:add(nn.SpatialBatchNormalization(64,1e-3))
bt6_1:add(cudnn.ReLU(true))
bt6_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt6_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt6_1:add(cudnn.ReLU(true))
bt6_1:add(nn.SpatialDropout(0.1))
bt6_2=nn.Sequential()
bt6_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt6_2:add(nn.SpatialBatchNormalization(32, 1e-3))
bt6_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck6:add(bt6_1)
bottleneck6:add(bt6_2)
SegModel:add(bottleneck6)
SegModel:add(nn.JoinTable(2,4))

bottleneck7=nn.ConcatTable()
bt7_1=nn.Sequential()
bt7_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt7_1:add(nn.SpatialBatchNormalization(64, 1e-3))
bt7_1:add(cudnn.ReLU(true))
bt7_1:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt7_1:add(nn.SpatialBatchNormalization(64,1e-3))
bt7_1:add(cudnn.ReLU(true))
bt7_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt7_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt7_1:add(cudnn.ReLU(true))
bt7_1:add(nn.SpatialDropout(0.1))
bt7_2=nn.Sequential()
bt7_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt7_2:add(nn.SpatialBatchNormalization(32, 1e-3))
bt7_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck7:add(bt7_1)
bottleneck7:add(bt7_2)
SegModel:add(bottleneck7)
SegModel:add(nn.JoinTable(2,4))

bottleneck8=nn.ConcatTable()
bt8_1=nn.Sequential()
bt8_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt8_1:add(nn.SpatialBatchNormalization(64, 1e-3))
bt8_1:add(cudnn.ReLU(true))
bt8_1:add(nn.SpatialDilatedConvolution(64,64,3,3,1,1,8,8,8,8))
bt8_1:add(nn.SpatialBatchNormalization(64,1e-3))
bt8_1:add(cudnn.ReLU(true))
bt8_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt8_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt8_1:add(cudnn.ReLU(true))
bt8_1:add(nn.SpatialDropout(0.1))
bt8_2=nn.Sequential()
bt8_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt8_2:add(nn.SpatialBatchNormalization(32, 1e-3))
bt8_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck8:add(bt8_1)
bottleneck8:add(bt8_2)
SegModel:add(bottleneck8)
SegModel:add(nn.JoinTable(2,4))

bottleneck9=nn.ConcatTable()
bt9_1=nn.Sequential()
bt9_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt9_1:add(nn.SpatialBatchNormalization(64, 1e-3))
bt9_1:add(cudnn.ReLU(true))
bt9_1:add(cudnn.SpatialConvolution(64,64,5,1,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt9_1:add(cudnn.SpatialConvolution(64,64,1,5,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt9_1:add(nn.SpatialBatchNormalization(64,1e-3))
bt9_1:add(cudnn.ReLU(true))
bt9_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt9_1:add(nn.SpatialBatchNormalization(32, 1e-3))
bt9_1:add(cudnn.ReLU(true))
bt9_1:add(nn.SpatialDropout(0.1))
bt9_2=nn.Sequential()
bt9_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
bt9_2:add(nn.SpatialBatchNormalization(32, 1e-3))
bt9_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
bottleneck9:add(bt9_1)
bottleneck9:add(bt9_2)
SegModel:add(bottleneck9)
SegModel:add(nn.JoinTable(2,4))


--------------------------------------------------------------------------
----------------------DECODER---------------------------------------------
--nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH], [adjW], [adjH])

dbottleneck1=nn.ConcatTable()
dbt1_1=nn.Sequential()
dbt1_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt1_1:add(nn.SpatialBatchNormalization(64, 1e-3))
dbt1_1:add(cudnn.ReLU(true))
dbt1_1:add(cudnn.SpatialFullConvolution(64,64,2,2,2,2):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}):noBias())
dbt1_1:add(nn.SpatialBatchNormalization(64, 1e-3))
dbt1_1:add(cudnn.ReLU(true))
dbt1_1:add(nn.SpatialBatchNormalization(64,1e-3))
dbt1_1:add(cudnn.ReLU(true))
dbt1_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt1_1:add(nn.SpatialBatchNormalization(32, 1e-3))
dbt1_1:add(cudnn.ReLU(true))
dbt1_1:add(nn.SpatialDropout(0.1))
dbt1_2=nn.Sequential()
dbt1_2:add(cudnn.SpatialFullConvolution(64,32,2,2,2,2):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}):noBias())
dbt1_2:add(nn.SpatialBatchNormalization(32, 1e-3))
dbt1_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
dbottleneck1:add(dbt1_1)
dbottleneck1:add(dbt1_2)
SegModel:add(dbottleneck1)
SegModel:add(nn.JoinTable(2,4))

dbottleneck2=nn.ConcatTable()
dbt2_1=nn.Sequential()
dbt2_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt2_1:add(nn.SpatialBatchNormalization(64, 1e-3))
dbt2_1:add(cudnn.ReLU(true))
dbt2_1:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt2_1:add(nn.SpatialBatchNormalization(64, 1e-3))
dbt2_1:add(cudnn.ReLU(true))
dbt2_1:add(nn.SpatialBatchNormalization(64,1e-3))
dbt2_1:add(cudnn.ReLU(true))
dbt2_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt2_1:add(nn.SpatialBatchNormalization(32, 1e-3))
dbt2_1:add(cudnn.ReLU(true))
dbt2_1:add(nn.SpatialDropout(0.1))
dbt2_2=nn.Sequential()
dbt2_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt2_2:add(nn.SpatialBatchNormalization(32, 1e-3))
dbt2_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
dbottleneck2:add(dbt2_1)
dbottleneck2:add(dbt2_2)
SegModel:add(dbottleneck2)
SegModel:add(nn.JoinTable(2,4))

SegModel:add(cudnn.SpatialFullConvolution(64,64,2,2,2,2):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}):noBias())
SegModel:add(nn.SpatialBatchNormalization(64, 1e-3))
SegModel:add(cudnn.ReLU(true))

dbottleneck3=nn.ConcatTable()
dbt3_1=nn.Sequential()
dbt3_1:add(cudnn.SpatialConvolution(64,64,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt3_1:add(nn.SpatialBatchNormalization(64, 1e-3))
dbt3_1:add(cudnn.ReLU(true))
dbt3_1:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt3_1:add(nn.SpatialBatchNormalization(64, 1e-3))
dbt3_1:add(cudnn.ReLU(true))
dbt3_1:add(nn.SpatialBatchNormalization(64,1e-3))
dbt3_1:add(cudnn.ReLU(true))
dbt3_1:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt3_1:add(nn.SpatialBatchNormalization(32, 1e-3))
dbt3_1:add(cudnn.ReLU(true))
dbt3_1:add(nn.SpatialDropout(0.1))
dbt3_2=nn.Sequential()
dbt3_2:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))
dbt3_2:add(nn.SpatialBatchNormalization(32, 1e-3))
dbt3_2:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))
dbottleneck3:add(dbt3_1)
dbottleneck3:add(dbt3_2)
SegModel:add(dbottleneck3)
SegModel:add(nn.JoinTable(2,4))

SegModel:add(cudnn.SpatialConvolution(64,32,1,1,1,1):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1}))


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
--graph.dot(net.fg, 'SegModel', 'SegModel')
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





