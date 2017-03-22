require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'


SegModel=nn.Sequential()
--module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH])
SegModel:add(cudnn.SpatialConvolution(3,5,7,7,1,1))
SegModel:add(nn.SpatialBatchNormalization(5, 1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(5,10,5,5,1,1))
SegModel:add(nn.SpatialBatchNormalization(10, 1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialConvolution(10,15,3,3,1,1))
SegModel:add(nn.SpatialBatchNormalization(15, 1e-3))
SegModel:add(cudnn.ReLU(true))
--nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH], [adjW], [adjH])
SegModel:add(cudnn.SpatialFullConvolution(15,20,3,3,1,1))
SegModel:add(nn.SpatialBatchNormalization(20, 1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialFullConvolution(20,25,5,5,1,1))
SegModel:add(nn.SpatialBatchNormalization(25, 1e-3))
SegModel:add(cudnn.ReLU(true))
SegModel:add(cudnn.SpatialFullConvolution(25,30,3,3,1,1))

SegModel:add(nn.SpatialBatchNormalization(30, 1e-3))
SegModel:add(cudnn.ReLU(true))

SegModel:add(cudnn.SpatialFullConvolution(30,32,1,1,1,1))

criterion=cudnn.SpatialCrossEntropyCriterion()

SegModel=SegModel:cuda()
criterion:cuda()

return{
SegModel,
criterion,	
} 