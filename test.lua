require 'image'
require 'paths'
require 'torch'
require 'cunn'
require 'cudnn'
function test_model(opt)
	collectgarbage()
	torch.setnumthreads(opt.threads)
	local myFile=hdf5.open(opt.hdf5_test_path)
	local data = myFile:read('/dataset'):all()
	--close the hdf5 file
	myFile:close()
	
	test=torch.Tensor()
	for key,value in pairs(data['test']) do
		test=torch.cat(test,value:float(),4)
		data['test'][key]=nil
	end
	data=nil
	test=test:permute(4,3,1,2):cuda()
	SegNet=torch.load(opt.model_path)
	SegNet=SegNet:cuda()
	pred=SegNet:forward(test)
	torch.save(image .. '_prediction' .. '.th',pred)




	-- for f in paths.files(opt.test_path) do
 --   		if f~='.' and f~='..' then
 --   		p=opt.test_path .. f
 --   		im=image.load(p)
 --   		im=im:cuda()
	-- 	print("Next up, forward pass for ",p)
	-- 	SegNet=torch.load(opt.model_path)
	-- 	SegNet=SegNet:cuda()
	-- 	pred=SegNet:forward(im)
	-- 	torch.save(image .. '_prediction' .. '.th',pred)
 --   		end

	-- end

end
