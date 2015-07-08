require 'nn'
require 'kttorch'

-- creating a lookuptable
wordDims = (#glovevec)[2]
lkptbl = nn.LookupTable(glovevec:size(1), wordDims)

-- adjust the weight of the models, set them to the glove/word2vec rep
for i =1, glovevec:size(1) do
  lkptbl.weight[i] = glovevec[i]
end

noutputs = 3 
nhiddens = wordDims 
-- initliazing the model
model = nn.Sequential()
--Adding fixed lookup to avoid re-weighting the word-vec embeddings
model:add(kttorch.ImmutableModule(lkptbl)) 
-- model:add(lkptbl)
-- Concatenating Mean and Sum's outputs
m1 = nn.Concat(1)
m1:add(nn.Mean(1))
m1:add(nn.Sum(1))
model:add(m1) -- Adding the concatenated output to the model
model:add(nn.Dropout())

model:add(nn.Linear(wordDims*2,nhiddens*2))
model:add(nn.ReLU())
model:add(nn.Dropout())

model:add(nn.Linear(nhiddens*2,nhiddens*2))
model:add(nn.ReLU())
model:add(nn.Dropout())
--[[
model:add(nn.Linear(wordDims,nhiddens))
model:add(nn.ReLU())
model:add(nn.Linear(wordDims,nhiddens))
model:add(nn.ReLU())
model:add(nn.Linear(wordDims,nhiddens))
model:add(nn.ReLU())
--]]
model:add(nn.Linear(nhiddens*2,noutputs))
model:add(nn.LogSoftMax())

print '==> Here is the model:'
print(model)
