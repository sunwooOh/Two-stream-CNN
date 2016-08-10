function load_model(n_classes)
	split_no = opt.spl

	-- build ConvNet
	parallel_model = nn.ParallelTable()
	model_path = 'models/'

	if opt.mod ~= 'nil' then
		TwoConv = torch.load (opt.mod)

		return nil, nil, TwoConv
	end

	if opt.smod ~= 'nil' then
		spatial = torch.load (opt.smod)
	else
		if opt.res == 1 then
			model_name = model_path .. 'resnet/resnet-101.t7'
			spatial = torch.load(model_name):cuda()

			spatial:remove (spatial:size())
			spatial:add (nn.Linear (2048, 101))

		else
			proto_name = model_path .. 'cuhk_action_spatial_vgg_16_deploy.prototxt'
			binary_name = model_path .. 'vgg_16_action_rgb_pretrain.caffemodel'
			
			spatial = loadcaffe.load (proto_name, binary_name, 'cudnn')
			spatial:remove (spatial:size())
			spatial:remove (spatial:size())
			spatial:remove (spatial:size())
			spatial:remove (spatial:size())
			spatial:remove (spatial:size())

			spatial:add (nn.Dropout (0.9))
			spatial:add (nn.Linear (4096, 4096))
			spatial:add (cudnn.ReLU(true))
			spatial:add (nn.Dropout (0.9))
			spatial:add (nn.Linear (4096, 101))
		end

		-- Reset weights
		-- method = 'xavier'
		-- temporal = require ('weight-init') (spatial, method)
		-- print ('Weights initialized: xavier')
	end

	if opt.tmod ~= 'nil' then
		temporal = torch.load (opt.tmod)
	else
		if opt.res == 1 then
			model_name = model_path .. 'resnet/resnet-101.t7'
			temporal = torch.load(model_name)
			temporal:remove(1)
			temporal:insert(nn.SpatialConvolution(20, 64, 7, 7, 2, 2, 3, 3), 1)

			temporal:remove(temporal:size())
			temporal:add (nn.Linear (2048, 101))

			temporal:cuda()
			print (temporal)

		else

			proto_name = model_path .. 'cuhk_action_temporal_vgg_16_flow_deploy.prototxt'
			binary_name = model_path .. 'vgg_16_action_flow_pretrain.caffemodel'
			-- binary_name = 'cuhk_action_temporal_vgg_16_split' .. split_no .. '.caffemodel'

			temporal = loadcaffe.load (proto_name, binary_name, 'cudnn')
			temporal:add (nn.Linear (4096, 101))
		end
		-- temporal:add (nn.LogSoftMax():cuda())

		-- Reset weights
		-- method = 'xavier'
		-- temporal = require ('weight-init') (temporal, method)
		-- print ('Weights initialized: xavier')

	end

	parallel_model:add (spatial)
	parallel_model:add (temporal)


	-- parallel_model:add (nn.CAddTable())
	-- parallel_model:cuda()

	TwoConv = nn.Sequential ()
	TwoConv:add (parallel_model)
	TwoConv:add (nn.CAddTable():cuda())
	TwoConv:add (nn.MulConstant (0.5, false):cuda())


	-- TwoConv:add (nn.Sum(1))
	-- TwoConv:add (nn.JoinTable(1))
	-- TwoConv:add (nn.LogSoftMax():cuda())

	return spatial, temporal, TwoConv
end
