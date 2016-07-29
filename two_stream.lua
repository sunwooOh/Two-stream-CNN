function two_stream (model)
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

	train_loss = {}
	train_loss_mean = {}
	train_acc = {}
	val_loss = {}
	val_acc = {}

	for e = 1, epochs do
		-- train:
		rand_lines = torch.randperm (tot_trains)
		rand_subl = rand_lines [{ { 1, math.min (tr_iter, tot_trains) } }]

		loss_train, acc_train = train (model, rand_subl, 23, e, target_tab, split_num)

		-- test:
		rand_lines = torch.randperm (tot_tests)
		rand_subl = rand_lines [{ { 1, math.min (te_iter, tot_tests) } }]

		acc_test = test (model, rand_subl, 23, e, target_tab, split_num)

	end

end
