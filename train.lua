function train (net, random_input_table, channel_num, epc, target_ind_tab)
	--[[
		net: 	model	
		channel num:	20 for optical flow
				3 for rgb image
				23 for two_stream

		inputs: table
		contents of the inputs: double tensor
	--]]

	print ('Current lrate:	' .. opt.lrate)

	split_num = opt.spl

	train_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0"
	train_path = train_path .. split_num .. ".txt"

	if channel_num == 20 then	-- Temporal net
		batch_size = opt.tbat
		_channel_num = 20
		-- lrate_decay = 0.1
		-- lrate_stop = 30000/batch_size
		-- lrate_step = 10000/batch_size
	elseif channel_num == 3 then
		batch_size = opt.bat
		_channel_num = 3
		--lrate_decay = 0.1
		--lrate_stop = 10000/batch_size
		--lrate_step = 4000/batch_size
	else
		batch_size = opt.twobat
		_channel_num = 20
	end

	clip = opt.tgc
	weight_decay = opt.twd
	learning_rate = opt.lrate
	max_iter = opt.titer
	epochs = opt.epc
	conf_mat = torch.Tensor (101, 101):zero ()

	loss_vals = {}
	acc_vals = {}

--	criterion = nn.ClassNLLCriterion():cuda()
	-- criterion = nn.CriterionTable (nn.CrossEntropyCriterion():cuda())
	criterion = nn.CrossEntropyCriterion():cuda()
	
	params, grad_params = net:getParameters ()

	t_inputs = torch.DoubleTensor (batch_size, _channel_num, 224, 224)
	t_sp_inputs = torch.DoubleTensor (batch_size, 3, 224, 224)
	t_targets = torch.DoubleTensor (batch_size)

	optim_state = {
		learningRate = learning_rate,
		-- weightDecay = weight_decay,
		-- momentum = 0.9
	}

	for i = 1, max_iter, batch_size do
		train_list = io.open (train_path, "r")

		rand_subl = random_input_table[{ { i, math.min(max_iter, i+batch_size-1) } }]
		rand_subl = torch.totable (rand_subl)
		table.sort (rand_subl, function (a, b) return a < b end)
		
		-- Load image & get frames
		if channel_num == 20 then
			inputs, targets = get_opflow (rand_subl, target_ind_tab, train_list)
		elseif channel_num == 3 then
			inputs, targets = get_video (rand_subl, train_list, target_ind_tab)
		else
			sp_inputs, targets = get_video (rand_subl, train_list, target_ind_tab)
			train_list = io.open (train_path, "r")
			tm_inputs, targets = get_opflow (rand_subl, target_ind_tab, train_list)
		end

		if channel_num ~= 23 then
			for bat = 1, batch_size do
				t_inputs[bat]:copy (inputs[bat])
				t_targets[bat] = targets[bat]

			end
			
			c_inputs = t_inputs:cuda ()
			c_targets = t_targets:cuda ()

		else		-- Two-stream cnn
			for bat = 1, batch_size do
				t_sp_inputs[bat]:copy (sp_inputs[bat])
				t_inputs[bat]:copy (tm_inputs[bat])
				t_targets[bat] = targets[bat]
			end

			t_sp_inputs = t_sp_inputs:cuda()
			t_inputs = t_inputs:cuda()
			c_inputs = {}
			table.insert (c_inputs, t_sp_inputs)
			table.insert (c_inputs, t_inputs)
			c_targets = t_targets:cuda()
		end	
	
		-- Net forward/backward ------------------------------------------------------
		local feval = function (params)
			grad_params:zero ()

			output = net:forward (c_inputs)
			-- if channel_num == 23 then
			-- 	output = output[1] + output[2]
			-- 	output:div(2)
			-- end
			-- print (output)

			loss = criterion:forward (output, c_targets)
			-- print (loss)
			
			dloss_dout = criterion:backward (output, c_targets)
			-- print (dloss_dout)
			net:backward (c_inputs, dloss_dout)	--
			
			-- Gradient clipping
			grad_params = grad_params:clamp (-clip, clip)

			-- L2 Regularization
			if weight_decay ~= 0 then
				table.insert (loss_vals, loss)
				loss = loss + weight_decay * torch.norm (params, 2)^2
				grad_params:add (params:clone ():mul (weight_decay))

				print ('Loss after L2 reg: ' .. loss)

			else
				table.insert (loss_vals, loss)
			end

			print ('-----------------------------------------------------')
			print ('   [[[ Batch ' .. counter .. ' / ' .. opt.epc*max_iter/batch_size .. ' -- Epoch '.. epc ..' / '..epochs..' ]]]')
			print ('Loss: ' .. loss)

			return loss, grad_params
		end
		------------------------------------------------------------------------------

		if isnan (loss) then
			print ('-++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-')
			print ('-++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-')
			print ('		 	NaN detected!!!!')
			print ('-++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-')
			print ('-++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-')
		end


		-- Ratio of weights:update ---------------------------------------------------
		-- Ratio: update / param
		update_scale = (grad_params*learning_rate):norm()
		param_scale = params:norm()
		------------------------------------------------------------------------------


		-- Vanilla update the weights ------------------------------------------------
--		params:add(grad_params:mul(-learning_rate))
		------------------------------------------------------------------------------
		
		
		-- SGD using optim -----------------------------------------------------------
		optim.sgd (feval, params, optim_state)
		------------------------------------------------------------------------------

		conf_mat, accuracy = measure_acc (conf_mat, output, targets, batch_size)

		-- table.insert (loss_vals, loss)
		table.insert (acc_vals, accuracy)

		print ('Ratio (updates / weights) = ' .. update_scale/param_scale)
		print ('Norm of grad_params: ' .. grad_params:norm())
		print ('Norm of params: ' .. params:norm())
		if channel_num < 23 then
			print ('Norm of inputs: ' .. c_inputs:norm())
		else
			print ('Norm of rgb: ' .. c_inputs[1]:norm())
			print ('Norm of optical flow: ' .. c_inputs[2]:norm())		
		end
		print ('Norm of outputs: ' .. output:norm())
		print ('-----------------------------------------------------')

--		if counter%100 == 0 then print (conf_mat) end

		counter = counter + 1

	end
	print (conf_mat)

	return loss_vals, accuracy, indices
end
