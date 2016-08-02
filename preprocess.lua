counter = 1

function get_frames (root_path, root_fname, channel, ext)
	--[[	num_channel ==	1 for temporal
				3 for spatial
		num_frames  ==	20 for temporal
				1 for spatial
	--]]

	-- original width and height
	width = 340
	height = 256
	depth = 3

	-- init_time = timer:time().real

	if channel == 1 then
		frame = torch.DoubleTensor (20, height, width)
		depth = 20
	end
	
	cnt = 0
	while true do
		if channel == 3 then
			file_name = root_path .. root_fname .. '_' .. cnt+1 .. ext
		elseif channel == 1 then
			file_name = get_name (root_path .. root_fname, cnt+1, 'jpg')
		end

		if paths.filep (file_name) then
			cnt = cnt + 1
		else
			break
		end
	end

	math.randomseed (socket.gettime (10000))
	if channel == 3 then
		num_frames = 1
		frm_idx = math.random (0, 10000)%(cnt-1)
	elseif channel == 1 then
		num_frames = 20
		frm_idx = math.random (0, 10000)%(cnt-20)
		frm_idx = 1
	end

	for f = 1, num_frames, 2 do
		if channel == 1 then
			fpath = get_name (root_path .. 'flow_x_', frm_idx+f, 'jpg')
			frame_x = image.load (fpath, channel, 'double')
			fpath = get_name (root_path .. 'flow_y_', frm_idx+f, 'jpg')
			frame_y = image.load (fpath, channel, 'double')
			frame[f]:copy(frame_x)
			frame[f+1]:copy(frame_y)
		
		elseif channel == 3 then
			fpath = root_path .. root_fname .. "_" .. frm_idx+f .. ext
			frame = image.load (fpath, channel, 'double')
			image.save ('spatial_in.png', frame)
		end
	end

	-- print ('frame size: ')
	-- print (frame:size())
	-- file = io.open ('input_sanity1.txt', 'w')
	-- TODO: normalization of the image --> after crop?

	corner = math.random (1, 5000) % 5
	multi = math.random (1, 5000) % 4 + 1
	flip = math.random (1, 1000) % 2
	scale = { 168, 192, 224, 256 }

	-- 1. corner & multi-scale crop to make 224 x 224 image
	if corner == 4 then
		ofs_x = (height-scale[multi])/2 + 1
		ofs_y = (width-scale[multi])/2 + 1
	else
		ofs_x = corner%2 * (height-scale[multi]) + 1--(1 - corner%2)
		ofs_y = math.floor (corner/2) * (width-scale[multi]) + 1--(1 - corner/2)
	end

	if channel == 3 then
		-- resize image : 240 x 320 to 256 x 320
		frame = image.scale (frame, width, height)
		frame = frame:resize (depth, height, width)
	end
	if isnan (frame:norm()) then
		print ('Input NaN detected at stage: AFTER CORNER CROP')
		print ('Root path: ' .. root_path)
		print ('Root fname: ' .. root_fname)
		print ('Frame index: ' .. frm_idx)
	end
	-- print ('frame size:')
	-- print (frame:size())
	
	-- -- corner crop / random flip / rgb jittering
	crp_frm = frame:narrow (2, ofs_x, scale[multi])
	crp_frm = crp_frm:narrow (3, ofs_y, scale[multi])

	-- print ('scale now : '..scale[multi])
	-- if scale[multi] == 224 then
	-- 	print ('if 224:')
	-- 	print (crp_frm:size())
	-- end

	if scale[multi] ~= 224 then
		crp_frm = image.scale (crp_frm, 224, 224)

		-- crp_frm = image.scale (frame, 224, 224)
		crp_frm = crp_frm:resize (depth, 224, 224)
	end

	if isnan (crp_frm:norm()) then
		print ('Input NaN detected at stage: AFTER SCALING')
		print ('Root path: ' .. root_path)
		print ('Root fname: ' .. root_fname)
		print ('Frame index: ' .. frm_idx)
	end

	assert (crp_frm:size(2) == 224)

	-- 2. random flip
	if flip == 1 then	-- flip image
		crp_frm = image.hflip (crp_frm)
	end
	if isnan (crp_frm:norm()) then
		print ('Input NaN detected at stage: AFTER HORIZONTAL FLIP')
		print ('Root path: ' .. root_path)
		print ('Root fname: ' .. root_fname)
		print ('Frame index: ' .. frm_idx)
	end

	-- print ('img mean before normalization: '..crp_frm:mean())
	-- print ('img std befor normalization: '..crp_frm:std())

	-- 3. normalization
	-- nm = crp_frm:norm()

	-- print ('image mean : ' .. crp_frm:mean())
	-- print ('image max : ' .. torch.max (crp_frm))
	-- print ('image min : ' .. torch.min (crp_frm))
	crp_frm:mul(255)

	for j = 1, depth do
		img_mean = crp_frm [{ {j}, {}, {} }]:mean()
		img_std = crp_frm [{ {j}, {}, {} }]:std()

		crp_frm [{ {j}, {}, {} }]:add(-img_mean)
		if isnan (crp_frm:norm()) then print ('img_mean') end

		-- if channel == 3 then
		-- 	crp_frm [{ {j}, {}, {} }]:div(img_std)
		-- 	if isnan (crp_frm:norm()) then print ('img_std') end
		-- end
	end

	if isnan (crp_frm:norm()) then
		print ('Input NaN detected at stage: AFTER NORMALIZATION')
		print ('Root path: ' .. root_path)
		print ('Root fname: ' .. root_fname)
		print ('Frame index: ' .. frm_idx)
		print ('crop_frm std: '..img_std)
		print ('image mean: ' .. img_mean)
	end
	-- print ("difference: ".. crp_frm:norm()-nm)

	-- rgb to bgr if spatial
	if channel == 3 then
		chan_r = crp_frm [{ {1}, {}, {} }]
		chan_g = crp_frm [{ {2}, {}, {} }]
		chan_b = crp_frm [{ {3}, {}, {} }]

		res_frm = torch.cat (torch.cat (chan_b, chan_g, 1), chan_r, 1)
	else
		res_frm = crp_frm
	end

	-- print ('[get_frames] time elapsed to preprocess image: ' .. timer:time().real-init_time)
	if isnan (res_frm:norm()) then
		print ('Input NaN detected at stage: AFTER rgb')
		print ('Root path: ' .. root_path)
		print ('Root fname: ' .. root_fname)
		print ('Frame index: ' .. frm_idx)
		print ('crop_frm std: '..img_std)
		print ('image mean: ' .. img_mean)
	end

	return res_frm
end

function get_video (linenum_tab, fname_file, target_tab)
	root_dir = "/data/sunwoo/ucf_spatial/"
	tmp_dir = "/home/sunwoo/Desktop/two-stream/frames_stored/"

	-- extract class name & file name from fname_tab
	line_cnt = 1
	tab_idx = 1

	inputs = {}
	targets = {}
	-- print (linenum_tab)

	while true do
		fname_line = fname_file:read ("*l")
		if not fname_line then
			print ("fname_line is nil")
		end

		if line_cnt == linenum_tab[tab_idx] then
			--[[ examples of filenames
				fname_line = WalkingWithDog/v_WalkingWithDog_g19_c04.avi 98
				cname = WalkingWithDog
				sname = g19
				fname = c04
			--]]
			string.gsub (fname_line, "(.-)/", function (a) cname = a end)
			string.gsub (fname_line, "_g(.-)_", function (a) sname = a end)
			string.gsub (fname_line .. '*', "_c(.-)%.avi", function (a) fname = a end)
			
			cname = cname
			sname = 'g' .. sname .. '/'
			fname = 'c' .. fname

			file_path = root_dir .. cname .. '/' .. sname
			frame = get_frames (file_path, fname, 3, '.png')

			table.insert (inputs, frame)
			target = target_tab[cname]
			table.insert (targets, target)

			tab_idx = tab_idx + 1
		end

		if tab_idx > #linenum_tab then break end
		line_cnt = line_cnt + 1
	end

	return inputs, targets
end

-- fetch data from extracted frames
function sp_preprocess (model, target_tab)
	split_num = opt.spl
	
	train_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0"
	train_path = train_path .. split_num .. ".txt"

	test_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/testlist0"
	test_path = test_path .. split_num .. ".txt"

	save_path = 'save_data/spatial/'
	
	tot_trains = 0
	for i in io.lines (train_path) do
		tot_trains = tot_trains + 1
	end

	tot_tests = 0
	for i in io.lines (test_path) do
		tot_tests = tot_tests + 1
	end

	-- pick training videos of a batch size randomly
	batch_size = opt.bat
	epochs = opt.epc
	epcs = {}
	tr_losses = {}
	tr_loss_nm = {}
	tr_accs = {}
	te_losses = {}
	te_accs = {}
	iter = {}
	lrstring = tostring (opt.lrate)
	s = 1

	for e = 1, epochs do
		train_list = io.open (train_path, "r")
		test_list = io.open (test_path, "r")

		max_iter = opt.titer

		-- train:
		rand_lines = torch.randperm (tot_trains)
		rand_subl = rand_lines [{ { 1, math.min (max_iter, tot_trains) } }]
		-- rand_subl = torch.totable(rand_subl)

		-- table.sort (rand_subl, function (a, b) return a < b end)

		-- get videos of the each sub list
		-- inputs, targets = get_video (rand_subl, train_list, target_tab)

		-- print ('input size in preprocess(): ' .. #inputs)

		print ("[preprocess.lua/preprocess] now calling train function")
		-- loss_train, acc_train = train (model, inputs, targets, 3)		-- net type 3 for rgb
		loss_train, acc_train = train (model, rand_subl, 3, e, target_tab)

		-- test: 
		max_iter = opt.eiter
		rand_lines = torch.randperm (tot_tests)

		rand_subl = rand_lines [{ { 1, math.min (max_iter, tot_tests) } }]
		-- rand_subl = torch.totable(rand_subl)

		-- table.sort (rand_subl, function (a, b) return a < b end)

		print ('[preprocess.lua/preprocess] now calling get_video for test set')
		
		-- get videos of the each sub list
		-- inputs, targets = get_video (rand_subl, test_list, target_tab)
	
		-- acc_test = test (model, inputs, targets, 3)
		acc_test = test (model, rand_subl, 3, e, target_tab)

		table.insert (epcs, e)
		table.insert (tr_accs, acc_train)
		table.insert (te_accs, acc_test)
		table.insert (tr_loss_nm, torch.Tensor(loss_train):mean())
		-- table.insert (te_losses, loss_test:norm())
		len = #iter

		for t = 1, #loss_train do
			if t > len then
				if isnan (loss_train[t]) then
					table.insert (tr_losses, -1)
				else
					table.insert (tr_losses, loss_train[t])
				end

				table.insert (iter, s)
				s = s + 1
			end
		end

		t_epcs = torch.Tensor (epcs)
		t_iter = torch.Tensor (iter)
		t_tr_accs = torch.Tensor (tr_accs)
		t_te_accs = torch.Tensor (te_accs)
		t_tr_losses = torch.Tensor (tr_losses)
		t_tr_loss_nm = torch.Tensor (tr_loss_nm)

		plot (t_epcs, t_tr_accs, 'Epoch', 'Accuracy (%)', 'Training Accuracy', 0)
		plot (t_epcs, t_te_accs, 'Epoch', 'Accuracy (%)', 'Validation Accuracy', 0)
		plot (t_iter, t_tr_losses, 'Iteration', 'Loss', 'Training Loss', 1)
		plot_mult (t_epcs, t_tr_accs, t_te_accs, 'Epoch', 'Training', 'Validation', 'Accuracy (%)', 'Training and Validation Accuracies')
		plot (t_epcs, t_tr_loss_nm, 'Epoch', 'Loss', 'Training Loss (epc)', 0)

		-- netsav = model:clone ('weight', 'bias')
		-- torch.save ('spatial_sp_' .. split_num .. '_' .. e .. '.t7', netsav)

		file_name = 'lr' .. lrstring .. 'bat' .. batch_size .. 'ti' .. opt.titer .. 'wd' .. opt.twd .. 'gc' .. opt.tgc
		if not paths.dirp (save_path .. file_name) then
			os.execute ('mkdir ' .. save_path .. file_name)
		end

		-- save tables
		tables = {
			tr_acc = t_tr_accs,
			vl_acc = t_te_accs,
			tr_loss = t_tr_losses,
			tr_loss_mean = t_tr_loss_nm,
			ten_it = t_iter,
			ten_ep = t_epcs
		}
		torch.save (save_path .. file_name .. '/tables.dat', tables)

		-- save current network
		netsav = model:clone ('weight', 'bias')
		torch.save (save_path .. file_name .. '/split_' .. split_num .. '_' .. e .. '.t7', netsav)

	end

	train_list:close ()
	test_list:close ()

end

function get_name (root_name, idx, ext)
	return root_name .. string.format ("%04d", idx) .. '.' .. ext
end

function get_opflow (linenum_tab, target_tab, fname_table)
	-- Get 20 consecutive frames randomly
	rpath = '/data/sunwoo/ucf101_flow_img_tvl1_gpu/'

	line_cnt = 1
	tab_idx = 1
	inputs = {}
	targets = {}

	while true do
		fname_line = fname_table:read ("*l")
		if not fname_line then
			print ("fname_line is nil")
			break
		end

		if line_cnt == linenum_tab[tab_idx] then
			--[[ examples of filenames
				fname_line = WalkingWithDog/v_WalkingWithDog_g19_c04.avi 98
				cname = WalkingWithDog
				sname = (g)19
				fname = (c)04
			--]]
			string.gsub (fname_line, "(.-)/", function (a) cname = a end)
			string.gsub (fname_line, "_g(.-)_", function (a) sname = a end)
			string.gsub (fname_line .. '*', "_c(.-)%.avi", function (a) fname = a end)
			
			sname = 'v_' .. cname .. '_g' .. sname .. '_c' .. fname .. '/'
			fpath = rpath .. cname .. '/' .. sname

			input_volume = get_frames (fpath, 'flow_x_', 1, '.jpg')
			target = target_tab[cname]

			table.insert (targets, target)
			table.insert (inputs, input_volume)

			tab_idx = tab_idx + 1
		end

		if tab_idx > #linenum_tab then break end
		line_cnt = line_cnt + 1
	end

	return inputs, targets
end

function tm_preprocess (model, target_tab)
	-- size of the original image: 1 x 256 x 340
	-- scale: probably [0, 1]

	split_num = opt.spl

	train_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0"
	train_path = train_path .. split_num .. ".txt"

	test_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/testlist0"
	test_path = test_path .. split_num .. ".txt"

	save_path = 'save_data/temporal/'
	
	tot_trains = 0
	for i in io.lines (train_path) do
		tot_trains = tot_trains + 1
	end

	tot_tests = 0
	for i in io.lines (test_path) do
		tot_tests = tot_tests + 1
	end

	max_iter = opt.titer
	batch_size = opt.tbat
	epochs = opt.epc
	lrstring = tostring (opt.lrate)

	epcs = {}
	tr_losses = {}
	tr_loss_nm = {}
	tr_accs = {}
	te_losses = {}
	te_accs = {}
	iter = {}
	s = 1
	
	lrate_decay = 0.1

	for e = 1, epochs do
		max_iter = opt.titer

		rand_lines = torch.randperm (tot_trains)
		rand_subl = rand_lines [{ { 1, math.min (max_iter, tot_trains) } }]

		print ("[preprocess.lua/tm_preprocess] now calling train function")
		time_prev = timer:time().real
		loss_train, acc_train = train (model, rand_subl, 20, e, target_tab)
		print ('[training] time elapsed: ' .. timer:time().real-time_prev)

		-- test: 
		max_iter = opt.eiter
		rand_lines = torch.randperm (tot_tests)
		rand_subl = rand_lines [{ { 1, math.min (max_iter, tot_tests) } }]

		print ('[preprocess.lua/tm_preprocess] now calling get_opflow for test set')
		time_prev = timer:time().real
		acc_test = test (model, rand_subl, 20, e, target_tab)
		print ('[testing] time elapsed: ' .. timer:time().real-time_prev)

		table.insert (epcs, e)
		table.insert (tr_accs, acc_train)
		table.insert (te_accs, acc_test)

		len = #iter

		for t = 1, #loss_train do
			if t > len then
				if isnan (loss_train[t]) then
					table.insert (tr_losses, -1)
				else
					table.insert (tr_losses, loss_train[t])
				end

				table.insert (iter, s)
				s = s + 1
			end
		end

		table.insert (tr_loss_nm, torch.Tensor(loss_train):mean())

		t_epcs = torch.Tensor (epcs)
		t_iter = torch.Tensor (iter)
		t_tr_accs = torch.Tensor (tr_accs)
		t_te_accs = torch.Tensor (te_accs)
		t_tr_losses = torch.Tensor (tr_losses)
		t_tr_loss_nm = torch.Tensor (tr_loss_nm)

		plot (t_epcs, t_tr_accs, 'Epoch', 'Accuracy (%)', 'Training Accuracy', 0)
		plot (t_epcs, t_te_accs, 'Epoch', 'Accuracy (%)', 'Validation Accuracy', 0)
		plot (t_iter, t_tr_losses, 'Iteration', 'Loss', 'Training Loss', 1)
		plot_mult (t_epcs, t_tr_accs, t_te_accs, 'Epoch', 'Training', 'Validation', 'Accuracy (%)', 'Training and Validation Accuracies')
		plot (t_epcs, t_tr_loss_nm, 'Epoch', 'Loss', 'Training Loss (epc)', 0)
		-- plot (torch.Tensor (epcs), torch.Tensor (te_losses), 'Epochs', 'Loss', 'test_loss')

		if e%4 == 3 and e < 15 then
			opt.lrate = opt.lrate*lrate_decay
			print ('------------------------------------------------------------')
			print ('	learning rate updated: ' .. opt.lrate)
			print ('------------------------------------------------------------')
		end

		file_name = 'lr' .. lrstring .. 'bat' .. batch_size .. 'ti' .. opt.titer .. 'wd' .. opt.twd .. 'gc' .. opt.tgc
		if not paths.dirp (save_path .. file_name) then
			os.execute ('mkdir ' .. save_path .. file_name)
		end

		-- save tables
		tables = {
			tr_acc = t_tr_accs,
			vl_acc = t_te_accs,
			tr_loss = t_tr_losses,
			tr_loss_mean = t_tr_loss_nm,
			ten_it = t_iter,
			ten_ep = t_epcs
		}
		torch.save (save_path .. file_name .. '/tables.dat', tables)

		-- save current network
		netsav = model:clone ('weight', 'bias')
		torch.save (save_path .. file_name .. '/split_' .. split_num .. '_' .. e .. '.t7', netsav)

	end

end
