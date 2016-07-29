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
		gnuplot.plot (x_val, y_val, '-')
	else
		gnuplot.plot (x_val, y_val)
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
		{ ylabel1, x_val, y_val1, '+-' },
		{ ylabel2, x_val, y_val2, '+-' }
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
