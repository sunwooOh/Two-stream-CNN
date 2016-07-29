require 'ffmpeg'

-- parse class names and indices
function parse_class(file_path)
	print (io.open (file_path, "r"))
	
	lines = {}
	for line in io.lines (file_path) do
		for k in string.gmatch (line, '%w+') do
			if not tonumber (k) then
				lines [#lines + 1] = k
			end
		end
	end

	-- make targets list
	target_tab = {}
	for i = 1, #lines do
		target_tab[lines[i]] = i
	end
	
	return lines, target_tab
end

function load_data ()
	split_no = opt.spl

	file_path = "/home/sunwoo/Desktop/two-stream/UCF-101/"
	save_path = "/data/sunwoo/ucf_spatial/"

	-- dump frames of the training set
	movie_name_path = "ucfTrainTestlist/testlist0"

	file = io.open (movie_name_path .. split_no .. ".txt", "r")

	for movie_path in file:lines () do
		string.gsub (movie_path, '(.-)%.avi', function (a) movie_path = a end)
		video = ffmpeg.Video (file_path .. movie_path .. '.avi')
	
		vid_ten = video:totensor (1)
		print (vid_ten:size())
		
		print (movie_path)
		string.gsub (movie_path, "_(.-)_", function (a) class_name = a end)
		string.gsub (movie_path, "_g(.-)_", function (a) dir_name = a end)
		string.gsub (movie_path .. '*', "_c(.-)*", function (a) file_name = a end)

		class_name = class_name .. '/'
		dir_name = 'g' .. dir_name .. '/'
		file_name = 'c' .. file_name
		if not paths.dir (save_path .. class_name) then
			print ('making directory.. ')
			os.execute ("mkdir " .. save_path .. class_name) 
		end

		for i = 1, vid_ten:size(1) do
			if not paths.dir (save_path .. class_name .. dir_name) then
				print ('making directory.. ')
				os.execute ("mkdir " .. save_path .. class_name .. dir_name)
			end
			frame = vid_ten:select (1, i)
			img = frame:resize (3, 240, 320)
			image.save (save_path .. class_name .. dir_name .. file_name .. '_' .. i .. '.png', img)
		end
	end

	file:close()
end
