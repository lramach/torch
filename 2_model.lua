require 'nn'
require 'kttorch'

-- creating a lookuptable
wordDims = (#glovevec)[2]
lkptbl = nn.LookupTable(glovevec:size(1), wordDims)

-- adjust the weight of the models, set them to the glove/word2vec rep
for i =1, glovevec:size(1) do
  -- print(glovevec[i])
  lkptbl.weight[i] = glovevec[i]
end


-- noutputs = 200
noutputs = 4 
nhiddens = wordDims 
-- initliazing the model
model = nn.Sequential()
--model:add(kttorch.FixedLookupTable(lkptbl))
model:add(lkptbl)
model:add(nn.Mean(1))
model:add(nn.Linear(wordDims,nhiddens))
model:add(nn.ReLU())
model:add(nn.Linear(nhiddens,noutputs))
model:add(nn.LogSoftMax())
-- model = nn.Sequential()
-- model:add(lkptbl)
-- model:add(nn.Mean(1))
-- model:add(nn.Linear(wordDims, noutputs))
-- model:add(nn.Tanh())
-- model:add(nn.Linear(noutputs,1))


----------------------------------------------------------------------
print '==> here is the model:'
print(model)
