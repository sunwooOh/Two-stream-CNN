function test (net, random_input_table, channel_num, epc, target_ind_tab)
	--[[
		channel_num:
		20 for optical flow
		3 for rgb image
		23 for two stream net
	--]]
	split_num = opt.spl

	test_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/testlist0"
	test_path = test_path .. split_num .. ".txt"

	print ('[test.lua] in test function')

	if channel_num == 20 then
		batch_size = opt.tbat
		_channel_num = 20
	elseif channel_num == 3 then
		batch_size = opt.bat
		_channel_num = 3
	else
		batch_size = opt.twobat
		_channel_num = 20
	end
	max_iter = opt.eiter

	conf_mat = torch.DoubleTensor (101, 101):zero ()
	accs = {}
	losses = {}
	-- criterion = nn.ClassNLLCriterion()

	t_inputs = torch.DoubleTensor (batch_size, _channel_num, 224, 224)
	t_sp_inputs = torch.DoubleTensor (batch_size, 3, 224, 224)
	t_targets = torch.DoubleTensor (batch_size)

	for i = 1, max_iter, batch_size do
		test_list = io.open (test_path, "r")

		rand_subl = random_input_table[{ { i, math.min(max_iter, i+batch_size-1) } }]
		rand_subl = torch.totable (rand_subl)
		table.sort (rand_subl, function (a, b) return a < b end)

		-- Load image & get frames
		if channel_num == 20 then
			inputs, targets = get_opflow (rand_subl, target_ind_tab, test_list)
		elseif channel_num == 3 then
			inputs, targets = get_video (rand_subl, test_list, target_ind_tab)
		else
			sp_inputs, targets = get_video (rand_subl, test_list, target_ind_tab)
			test_list = io.open (test_path, "r")
			tm_inputs, targets = get_opflow (rand_subl, target_ind_tab, test_list)
		end

		if channel_num ~= 23 then
			for bat = 1, batch_size do
				t_inputs[bat]:copy (inputs[bat])
				t_targets[bat] = targets[bat]
			end
			c_inputs = t_inputs:cuda ()
			c_targets = t_targets:cuda ()
		else
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

		output = net:forward (c_inputs)
		-- loss = criterion:forward (output, t_targets)

		conf_mat, accuracy = measure_acc (conf_mat, output, targets, batch_size)

		table.insert (accs, accuracy)
		-- table.insert (losses, loss)
	end

	-- return torch.Tensor (losses), accuracy
	print (conf_mat)

	return accuracy
end
