
function load_model(n_classes)
	-- build ConvNet
	ConvNet = nn.Concat(2)

	-- spatial stream ConvNet
--	proto_name = 'models/bvlc_alexnet/train_val.prototxt'
--	binary_name = 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

	proto_name = 'cuhk_action_spatial_vgg_16_deploy.prototxt'
	binary_name = 'cuhk_action_spatial_vgg_16_split1.caffemodel'

	--proto_name = 'SqueezeNet/SqueezeNet_v1.1/train_val.prototxt'
	--binary_name = 'SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel'
	spatial = loadcaffe.load (proto_name, binary_name, 'cudnn')

	spatial:remove (spatial:size()-1)
	spatial:remove (spatial:size())
	spatial:add (nn.Linear (4096, n_classes))
	spatial:add (nn.LogSoftMax())

	-- temporal stream ConvNet
	temporal = spatial:clone()

	ConvNet:add(spatial)
	ConvNet:add(temporal)

	print (ConvNet)
	
	return ConvNet
end
