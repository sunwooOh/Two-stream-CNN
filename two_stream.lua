function two_stream (model)
	split_num = opt.spl

	train_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0"
	train_path = train_path .. split_num .. ".txt"

	test_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/testlist0"
	test_path = test_path .. split_num .. ".txt"

	save_path = 'save_data/two_stream/'
	
	tot_trains = 0
	for i in io.lines (train_path) do
		tot_trains = tot_trains + 1
	end

	tot_tests = 0
	for i in io.lines (test_path) do
		tot_tests = tot_tests + 1
	end

	tr_iter = opt.titer
	te_iter = opt.eiter
	sp_batch_size = opt.bat
	tm_batch_size = opt.tbat
	epochs = opt.epc
	learning_rate = opt.lrate
	lrstring = tostring (learning_rate)
	lrate_decay = 0.1
	split_num = opt.spl

	tr_losses = {}
	tr_loss_mean = {}
	tr_accs = {}
	te_losses = {}
	te_accs = {}
	iter = {}
	s = 0

	for e = 1, epochs do
		-- train:
		rand_lines = torch.randperm (tot_trains)
		rand_subl = rand_lines [{ { 1, math.min (tr_iter, tot_trains) } }]

		loss_train, acc_train = train (model, rand_subl, 23, e, target_tab, split_num)

		-- test:
		rand_lines = torch.randperm (tot_tests)
		rand_subl = rand_lines [{ { 1, math.min (te_iter, tot_tests) } }]

		acc_test = test (model, rand_subl, 23, e, target_tab, split_num)

		len = #iter
		for t = 1, #loss_train do
			if t > len then
				if isnan (loss_train[t]) or loss_train[t] > 30 then
					table.insert (tr_losses, -1)
				else
					table.insert (tr_losses, loss_train[t])
				end
				s = s + 1
				table.insert (iter, s)
			end
		end
	
		table.insert (tr_accs, acc_train)
		table.insert (te_accs, acc_test)
		table.insert (tr_loss_mean, torch.Tensor (loss_train):mean())

		t_iter = torch.Tensor (iter)
		t_tr_accs = torch.Tensor (tr_accs)
		t_te_accs = torch.Tensor (te_accs)
		t_tr_losses = torch.Tensor (tr_losses)
		t_tr_loss_mean = torch.Tensor (tr_loss_mean)

		plot (nil, t_tr_accs, 'Epoch', 'Accuracy (%)', 'Training Accuracy', 0)
		plot (nil, t_te_accs, 'Epoch', 'Accuracy (%)', 'Validation Accuracy', 0)
		plot (nil, t_tr_losses, 'Iteration', 'Loss', 'Training Loss', 1)
		plot_mult (nil, t_tr_accs, t_te_accs, 'Epoch', 'Training', 'Validation', 'Accuracy (%)', 'Training and Validation Accuracies')
		plot (nil, t_tr_loss_mean, 'Epoch', 'Loss', 'Training Loss (per epoch)', 0)

		-- Learning rate decay
		-- if e%4 == 0 and e < 15 then
		-- 	opt.lrate = opt.lrate * lrate_decay
		-- 	print ('----------------------------------------------------------------')
		-- 	print ('	learning rate updated: ' .. opt.lrate)
		-- 	print ('----------------------------------------------------------------')
		-- end

		file_name = 'lr' .. lrstring .. 'bat' .. sp_batch_size .. 'ti' .. opt.titer .. 'wd' .. opt.twd .. 'gc' .. opt.tgc

		if not paths.dirp (save_path .. file_name) then
			os.execute ('mkdir ' .. save_path .. file_name)
		end

		netsav = model:clone ('weight', 'bias')
		torch.save (save_path .. file_name .. '/split_' .. split_num .. '_' .. e .. '.t7', netsav)
	end

end
