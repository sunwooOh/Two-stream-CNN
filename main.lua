require 'paths'
require 'loadcaffe'
require 'nn'
require 'cunn'
require 'cudnn'		-- gpu mode
require 'image'		-- rescaling
require 'optim'		-- confusion matrix, sgd

paths.dofile ('models.lua')
paths.dofile ('dataset.lua')
paths.dofile ('preprocess.lua')

-- parse command line options
--[[
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-gpu', 0, 'do not use gpu')
cmd:text()
opt = cmd:parse(arg)
]]--

class_name_path = "ucfTrainTestlist/classInd.txt"
classes = parse_class(class_name_path)
print (classes)

-- TODO: splits 1,2&3 separately
--s1_ConvNet, s2_ConvNet, s3_ConvNet = load_model (#classes)

-- load datasets ILSVRC and UCF
--load_data ()

sp_train1, sp_test1 = sp_preprocess (1)

--[[
tm_train1, tm_test1 = tm_preprocess (1)

sp_train2, sp_test2 = sp_preprocess (2)
tm_train2, tm_test2 = tm_preprocess (2)

sp_train3, sp_test3 = sp_preprocess (3)
tm_train3, tm_test3 = tm_preprocess (3)
--]]

-- finetune on UCF-101 (actual training on spatial net)

-- test
