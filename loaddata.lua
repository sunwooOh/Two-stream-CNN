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
	
	return lines
end

function load_data (net)
	file_path = "/home/sunwoo/Desktop/two-stream/UCF-101/"
	save_path = "/home/sunwoo/Desktop/two-stream/frames_stored/"

	-- dump frames of the training set
	movie_name_path = "ucfTrainTestlist/trainlist0"

	for i = 1, 3 do	-- for training lists 1, 2 & 3
		print (io.open (movie_name_path .. i .. ".txt"), "r")
		for movie_path in io.lines (movie_name_path .. i .. ".txt") do
			string.gsub (movie_path, '(.-).avi', function (a) movie_path = a end)
			video = ffmpeg.Video (file_path .. movie_path .. '.avi')
		
			vid_ten = video:totensor (1)
			print (vid_ten:size())
			
			print (movie_path)
			string.gsub (movie_path, "_(.-)_", function (a) class_name = a end)
			string.gsub (movie_path, "g(.-)_", function (a) dir_name = a end)
			string.gsub (movie_path .. '*', "c(.-)*", function (a) file_name = a end)

			class_name = class_name .. '/'
			dir_name = 'g' .. dir_name .. '/'
			file_name = 'c' .. file_name
			print (class_name)
			print (dir_name)
			print (file_name)
			
			if not paths.dir (save_path .. class_name) then
				os.execute ("mkdir " .. save_path .. class_name) 
			end
	
			for i = 1, vid_ten:size(1) do
				if not paths.dir (save_path .. class_name .. dir_name) then
					os.execute ("mkdir " .. save_path .. class_name .. dir_name)
				end
				frame = vid_ten:select (1, i)
				img = frame:resize (3, 240, 320)
				image.save (save_path .. class_name .. dir_name .. file_name .. '_' .. i .. '.png', img)
			end
			-- save frames as an image
			-- just read files to get frames
			-- or get frames every time the video is read in?
		end
	end

	-- batch-ize training set

	-- TODO: if frames already dumped, don't do it again

end
