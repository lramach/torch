require 'torch'
--Load the model
require 'nn'
model = torch.load('results/model.net')

--Load the test data
dofile '1_load_test_data.lua'

--Load the test function
dofile '5_test_only.lua'
--Run test
print '===> Testing!'
test()
