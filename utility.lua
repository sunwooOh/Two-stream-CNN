function measure_acc (mat, output, targets, batch_size)		-- mat : 10 x 10

	for i = 1, batch_size do
		max = -math.huge
		for ind = 1, 101 do 	-- for each class
			if output[i][ind] > max then
				max_ind = ind
				max = output[i][ind]
			end
		end

		mat[targets[i]][max_ind] = mat[targets[i]][max_ind] + 1
	end

	correct = 0
	for i = 1, 101 do
		correct = correct + mat[i][i]
	end
	global_correct = correct / mat:sum() * 100

	print ('Accuracy : ' .. global_correct)

	return mat, global_correct

end

function plot (x_val, y_val, xlabel, ylabel, _title, line)
	fpath = 'plots/'
	file_name = 'lr' .. learning_rate .. 'bat' .. batch_size .. 'ti' .. opt.titer .. 'wd' .. opt.twd .. 'gc' .. opt.tgc ..'/' 
	if not paths.dirp (fpath .. file_name) then
		os.execute ('mkdir ' .. fpath .. file_name)
	end
	fname = _title:gsub("%s+", "") .. '.png'

	print ('[utility.lua/plot] Plotting ' .. fpath..fname.. ' with title '.._title)

	p = gnuplot.pngfigure (fpath .. file_name .. fname)

	gnuplot.grid (true)
	gnuplot.title (_title)
	gnuplot.xlabel (xlabel)
	gnuplot.ylabel (ylabel)

	if line == 1 then
		gnuplot.plot (y_val, '-')
	else
		gnuplot.plot (y_val)
	end
	
	gnuplot.plotflush ()
	gnuplot.close (p)

end

function plot_mult (x_val, y_val1, y_val2, xlabel, ylabel1, ylabel2, ylabel, _title)
	fpath = 'plots/'
	file_name = 'lr' .. learning_rate .. 'bat' .. batch_size .. 'ti' .. opt.titer .. 'wd' .. opt.twd .. 'gc' .. opt.tgc ..'/'
	if not paths.dirp (fpath .. file_name) then
		os.execute ('mkdir ' .. fpath .. file_name)
	end
	fname = _title:gsub("%s+", "") .. '.png'

	print ('[utility.lua/plot] Plotting ' .. fpath..fname.. ' with title '.._title)

	p = gnuplot.pngfigure (fpath .. file_name .. fname)

	gnuplot.grid (true)
	gnuplot.title (_title)
	gnuplot.xlabel (xlabel)
	gnuplot.ylabel (ylabel)

	gnuplot.plot (
		{ ylabel1, y_val1, '+-' },
		{ ylabel2, y_val2, '+-' }
	)

	gnuplot.plotflush ()
	gnuplot.close (p)

end

function isnan (number)
	return number ~= number
end

function save_net (model, path)
	netsav = model:clone ('weight', 'bias')
	torch.save (path, model)
end

function mean_calc ()
	for split_no = 1, 3 do
		write_to = torch.DiskFile ('per_pixel_mean' .. split_no .. '.txt', 'w')

		root_dir = "/data/sunwoo/ucf_spatial/"
		train_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0"
		train_path = train_path .. split_no .. ".txt"

		-- train_list = io.open (train_path, 'r')

		sum = 0
		cnt = 0

		for fline in io.lines (train_path) do
			string.gsub (fline, "(.-)/", function (a) cname = a end)
			string.gsub (fline, "_g(.-)_", function (a) sname = a end)
			string.gsub (fline .. '*', "_c(.-)%.avi", function (a) fname = a end)
			
			cname = cname
			sname = 'g' .. sname .. '/'
			fname = 'c' .. fname

			file_path = root_dir .. cname .. '/' .. sname
			-- frame = get_frames (file_path, fname, 3, '.png')

			idx = 1
			while true do
				-- file_name = root_path .. root_fname .. '_' .. idx+1 .. ext
				file_name = file_path .. fname .. '_' .. idx .. '.png'
				print (file_name)

				if paths.filep (file_name) then
					fpath = root_path 
					img = image.load (file_name, 3, 'double')
					-- img:mul(255)
					print (img:mean())
					print (img:sum())
					idx = idx + 1
					sum = sum + img:sum()
					print (sum)
					cnt = cnt + 1
				else
					break
				end
			end

			
		end

		write_to:writeObject ('sum: ' .. sum)
		write_to:writeObject ('count: ' ..cnt)
		write_to:writeObject ('Per pixel mean of split ' .. split_no .. ': ' .. 255*sum/cnt)
		write_to:close()
	end
end
function mean_calc_tmp ()
	for split_no = 1, 3 do
		write_to = torch.DiskFile ('tmp_mean' .. split_no .. '.txt', 'w')

		root_dir = '/data/sunwoo/ucf101_flow_img_tvl1_gpu/'
		train_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0"
		train_path = train_path .. split_no .. ".txt"

		-- train_list = io.open (train_path, 'r')

		sum = 0
		cnt = 0

		for fline in io.lines (train_path) do
			string.gsub (fline, "(.-)/", function (a) cname = a end)
			string.gsub (fline, "_g(.-)_", function (a) sname = a end)
			string.gsub (fline .. '*', "_c(.-)%.avi", function (a) fname = a end)
			
			sname = 'v_' .. cname .. '_g' .. sname .. '_c' .. fname .. '/'
			fpath = rpath .. cname .. '/' .. sname

			file_path = root_dir .. cname .. '/' .. sname
			-- frame = get_frames (file_path, fname, 3, '.png')

			idx = 1
			while true do
				-- file_name = root_path .. root_fname .. '_' .. idx+1 .. ext
				-- file_name = file_path .. fname .. '_' .. idx .. '.jpg'
				fpath_x = get_name (file_path .. fname .. 'flow_x_', idx, 'jpg')
				fpath_y = get_name (file_path .. fname .. 'flow_y_', idx, 'jpg')

				print (fpath)

				if paths.filep (file_name) then
					fpath = root_path 
					img_x = image.load (fpath_x, 1, 'double')
					print (img:mean())
					print (img:sum())
					idx = idx + 1
					sum = sum + img:sum()
					print (sum)
					cnt = cnt + 1
				else
					break
				end
			end

			
		end

		write_to:writeObject ('sum: ' .. sum)
		write_to:writeObject ('count: ' ..cnt)
		write_to:writeObject ('Per pixel mean of split ' .. split_no .. ': ' .. 255*sum/cnt)
		write_to:close()
	end
end