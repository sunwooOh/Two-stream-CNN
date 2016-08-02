function train2 (net, random_input_table, channel_num, epc, target_ind_tab)
	split_num = opt.spl

	train_path = '/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0'
	train_path = train_path .. split_num .. '.txt'

	batch_size = opt.tbat
	clip = opt.tgc
	weight_decay = opt.twd
	learning_rate = opt.lrate
	max_iter = opt.titer
	epochs = opt.epc

	conf_mat = torch.Tensor (101, 101):zero()
	loss_vals = {}
	acc_vals = {}

	optim_state = {
		learningRate = learning_rate,
		weightDecay = weight_decay
	}

	criterion = nn.CrossEntropyCriterion():cuda()

	params, grad_params = net:getParameters ()

	tensor_inputs = torch.DoubleTensor (batch_size, channel_num, 224, 224)
	tensor_targets = torch.DoubleTensor (batch_size)

	for i = 1, max_iter, batch_size do
		train_list = io.open (train_path, 'r')
		rand_subl = random_input_table [{ { i, math.min (max_iter, i+batch_size-1) } }]
		rand_subl = torch.totable (rand_subl)
		table.sort (rand_subl, function (a, b) return a < b end)

		-- Load frames (inputs)
		inputs, targets = get_opflow (rand_subl, target_ind_tab, train_list)

		for bat = 1, batch_size do
			tensor_inputs[bat]:copy (inputs[bat])
			tensor_targets[bat] = targets[bat]
		end

		cdtensor_inputs = tensor_inputs:cuda()
		cdtensor_targets = tensor_targets:cuda()

		-- local feval = function (params)
			grad_params:zero()

			output = net:forward (cdtensor_inputs)
			loss = criterion:forward (output, cdtensor_targets)

			dloss_dout = criterion:backward (output, cdtensor_targets)
			net:backward (cdtensor_inputs, dloss_dout)

			grad_params = grad_params:clamp (-clip, clip)

			table.insert (loss_vals, loss)

		-- 	return loss, grad_params
		-- end

		update_scale = (grad_params*learning_rate):norm()
		param_scale = params:norm()

		-- optim.sgd (feval, params, optim_state)
		params:add(grad_params:mul(-learning_rate))

		conf_mat, accuracy = measure_acc (conf_mat, output, targets, batch_size)

		table.insert (acc_vals, accuracy)

		train_list:close()

		 print ('   [[[ tr2 Batch ' .. counter .. ' / ' .. opt.epc*max_iter/batch_size     .. ' -- Epoch '.. epc ..' / '..epochs..' ]]]')
		 print ('Loss: '..loss)
		 print ('Ratio (updates / weights) = ' .. update_scale/param_scale)
         print ('Norm of grad_params: ' .. grad_params:norm())
         print ('Norm of params: ' .. params:norm())
         print ('Norm of inputs: ' .. cdtensor_inputs:norm())
         
         print ('Norm of outputs: ' .. output:norm())
         print ('-----------------------------------------------------')

         counter = counter + 1
	end

	print (conf_mat)

	return loss_vals, accuracy
end