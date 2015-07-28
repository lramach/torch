require 'nn'
require 'kttorch'

-- creating a lookuptable
wordDims = (#glovevec)[2]
lkptbl = nn.LookupTable(glovevec:size(1), wordDims)

-- adjust the weight of the models, set them to the glove/word2vec rep
for i =1, glovevec:size(1) do
  lkptbl.weight[i] = glovevec[i]
end
print(lkptbl.weight:size())
noutputs = opt.numscores 
nhiddens = wordDims 
m1 = nn.Sequential()
--Initializing the FMeasMatch class
m1:add(kttorch.FMeasMatch(references)) 
-- Produce a number as the output

-- Generate sum of the glove vec embeddings
m2 = nn.Sequential()
m2:add(kttorch.ImmutableModule(lkptbl))
m2:add(nn.Sum(1))

-- Concat the output from the above two models
model = nn.Sequential()
model:add(nn.Linear(wordDims+1,nhiddens*2))
model:add(nn.ReLU())
model:add(nn.Dropout())
model:add(nn.Linear(nhiddens*2,nhiddens*2))
model:add(nn.ReLU())
model:add(nn.Dropout())
model:add(nn.Linear(nhiddens*2,noutputs))
model:add(nn.LogSoftMax())

print '==> Here is the model:'
print(model)
