function get_frames (root_path, root_fname)
	frm_idx = 1
	one_clip = {}
	while true do
		fpath = root_path .. root_fname .. "_" .. frm_idx .. ".png"
		if paths.filep (fpath) then
			frame = image.load (fpath, 3, 'byte')
			one_clip[frm_idx] = frame
			frm_idx = frm_idx + 1
		else
			break
		end
	end

	return one_clip
end

function get_video (linenum_tab, fname_file)
	root_dir = "/data/sunwoo/ucf_spatial/"
	tmp_dir = "/home/sunwoo/Desktop/two-stream/frames_stored/"
	-- extract class name & file name from fname_tab
	line_cnt = 1
	tab_idx = 1

	minibatch = {}
	print (linenum_tab)
	while true do
		fname_line = fname_file:read ("*l")
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
			
			cname = cname .. '/'
			sname = 'g' .. sname .. '/'
			fname = 'c' .. fname

--			file_path = root_dir .. cname .. sname
			file_path = tmp_dir .. cname .. sname
			minibatch[tab_idx] = get_frames (file_path, fname)
			tab_idx = tab_idx + 1
		end

		print (table.getn(minibatch))
		
		if tab_idx > 256 then break end
		line_cnt = line_cnt + 1
	end

	print (table.getn(minibatch))
end

-- fetch data from extracted frames
function sp_preprocess (set_num)
	train_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/trainlist0"
	train_path = train_path .. set_num .. ".txt"

	test_path = "/home/sunwoo/Desktop/two-stream/ucfTrainTestlist/testlist0"
	test_path = test_path .. set_num .. ".txt"
	
	tot_trains = 0
	for i in io.lines (train_path) do
		tot_trains = tot_trains + 1
	end

	tot_tests = 0
	for i in io.lines (test_path) do
		tot_tests = tot_tests + 1
	end

	train_list = io.open (train_path, "r")
	test_list = io.open (test_path, "r")

	-- pick 256 training videos randomly
	rand_vids = {}		-- random video lines as entries

	-- TODO: lines -> a set of "multiple" subl's
	rand_lines = torch.randperm (tot_trains)
	rand_subl = rand_lines[{ {1, 256} }]
	rand_subl = torch.totable(rand_subl)

	table.sort (rand_subl, function (a, b) return a < b end)

	-- get videos of the each sub list
	rand_vids = get_video (rand_subl, train_list)


	train_list:close ()
	test_list:close ()

	-- resize/random crop/random flip/RGB jittering
	

	-- RGB to BGR?

	-- organize frames into batches
	-- (batch size: 256)



end

function tm_preprocess (set_num)

end
