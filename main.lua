require 'paths'
require 'loadcaffe'
require 'nn'
require 'cunn'
require 'cudnn'		-- gpu mode
require 'image'		-- rescaling, save, load
require 'socket'	-- randomseed
require 'gnuplot'
require 'optim'		-- solver

paths.dofile ('models.lua')
paths.dofile ('dataset.lua')
paths.dofile ('preprocess.lua')
paths.dofile ('train.lua')
paths.dofile ('test.lua')
paths.dofile ('utility.lua')
paths.dofile ('two_stream.lua')
paths.dofile ('train_v2.lua')
-- Parse command line options
cmd = torch.CmdLine()
cmd:text ()
cmd:text ('Options:')
cmd:option ('-bat', 30, 'size of a minibatch for spatial net')
cmd:option ('-tbat', 5, 'size of a minibatch for temporal net')
cmd:option ('-twobat', 4, 'size of a minibatch for two_stream net')
cmd:option ('-epc', 60, 'number of epochs')
cmd:option ('-lrate', 0.005, 'learning rate')
cmd:option ('-titer', 9510, 'number of iterations per a training pass')
cmd:option ('-eiter', 3690, 'number of iterations per an evaluation')
cmd:option ('-smod', 'nil', 'path of the trained spatial model to be loaded')
cmd:option ('-tmod', 'nil', 'path of the trained temporal model to be loaded')
cmd:option ('-twd', 0, 'LAMBDA value(weight decay) of L2-regularization for temporal net')
cmd:option ('-tgc', 1000, 'gradient clipping')
cmd:option ('-spl', 1, 'split number')
cmd:text ()
opt = cmd:parse(arg)

class_name_path = "ucfTrainTestlist/classInd.txt"
classes, target_tab = parse_class(class_name_path)

-- Start Timer
timer = torch.Timer ()

-- TODO:  splits 1,2&3 separately
spatial, temporal, ConvNet = load_model (#classes)

-- print ('[loading model] time elapse: ' .. timer:time().real)

-- print ('Entire ConvNet')
print (ConvNet)

-- Train spatial and temporal nets respectively
-- sp_preprocess (spatial, target_tab)
tm_preprocess (temporal, target_tab)

-- two_stream (ConvNet)

--[[
load_data(1)
load_data(2)
load_data(3)
--]]
