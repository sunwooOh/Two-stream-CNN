require 'paths'
require 'loadcaffe'
require 'nn'
require 'cunn'
require 'cudnn'		-- gpu mode
require 'image'		-- rescaling
require 'optim'		-- confusion matrix, sgd

paths.dofile ('loadmodels.lua')
paths.dofile ('loaddata.lua')

-- parse command line options
--[[
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-gpu', 0, 'do not use gpu')
cmd:text()
opt = cmd:parse(arg)
]]--

n_classes = 101

ConvNet = load_model()

-- load datasets ILSVRC and UCF
trainset, testset = load_data()

-- finetune on UCF-101
