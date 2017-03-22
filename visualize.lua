require 'nn'
require 'cudnn'
require 'cunn'
require 'hdf5'
require 'torch'
require 'optim'
require 'nngraph'
require 'image'


version=7
samples=5
classes=32
pred=torch.load('./test/prediction.th')

for i=1,samples do
	for j=1,classes do
		t=pred[{{i},{j},{},{}}]
		t=torch.exp(t)
		t=t:resize(1,180,240)
	image.save('./output_images/output_v'.. version .. '_' .. tostring(i) .. '_' .. tostring(j) .. '.jpg',t)	
	end
end


