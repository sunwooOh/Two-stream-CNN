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
	
	-- dump frames of the training set
	movie_name_path = "ucfTrainTestlist/trainlist0"

	for i = 1, 3 do
		print (io.open (movie_name_path .. i .. ".txt"), "r")
		for movie_path in io.lines (movie_name_path .. i .. ".txt") do
			for k in string.gmatch (movie_path, '%w+') do
				if not tonumber (k) then
					movie_path = k
				end
			end
--			movie_path = file_path .. movie_path:sub (1, movie_path:len()-3)
			video = ffmpeg.Video (movie_path)
		
			video:dump ("frames")	
		end
	end

	-- batch-ize training set

	-- TODO: if frames already dumped, don't do it again

end
