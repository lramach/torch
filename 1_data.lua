require 'hdf5';

-- reading in the glove vectors (word-embeddings)
myFile = hdf5.open('../../data/glovevec.'.. opt.prompt ..'.h5', 'r')
df = myFile:read():all()
glovevec = df.glovevec

-- reading the input, contains the indices of the words
myFile = hdf5.open('../../data/'.. opt.prompt ..'_train.h5', 'r')
df = myFile:read():all()
trainInput = df.index 
trainLabels = df.target +1 
--torch.cat(torch.Tensor(10):fill(1), torch.Tensor(10):fill(2)) 

--reading the validation dataset
myFile = hdf5.open('../../data/' .. opt.prompt ..'_validation.h5', 'r')
df = myFile:read():all()
validationInput = df.index
validationLabels = df.target + 1
print '==> loading dataset'

--Loading train data
trainData = {
  data = trainInput,
  labels = trainLabels, 
  size = function() return trainInput:size(1) end
}

--Loading validation data
validationData = {
  data = validationInput, 
  labels = validationLabels,
  size = function() return validationInput:size(1) end
}
trsize = (#trainInput)[1]
vasize = (#validationInput)[1]
