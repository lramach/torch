require 'torch'
--Load the model
require 'nn'
model = torch.load('results/model.net')

--Load the test data
dofile 'load_test_data.lua'

--Load the test function
dofile 'test_only.lua'
--Run test
print '===> Testing!'
test()
