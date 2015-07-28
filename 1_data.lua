require 'hdf5';

-- prompt = "COSC120160"
-- reading in the glove vectors (word-embeddings)
myFile = hdf5.open('pathToDataglovevec.'.. opt.prompt ..'.h5', 'r')
df = myFile:read():all()
glovevec = df.glovevec

-- reading the input, contains the indices of the words
myFile = hdf5.open('pathToData'.. opt.prompt ..'_train.h5', 'r')
df = myFile:read():all()
trainInput = df.index 
trainLabels = df.target +1 
--torch.cat(torch.Tensor(10):fill(1), torch.Tensor(10):fill(2)) 

--reading the validation dataset
myFile = hdf5.open('pathToData' .. opt.prompt ..'_validation.h5', 'r')
df = myFile:read():all()
validationInput = df.index
validationLabels = df.target + 1

--reading the references dataset
myFile = hdf5.open('pathToData' .. opt.prompt ..'_references.h5', 'r')
df = myFile:read():all()
referencesInput = df.index

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

--Loading the reference vectors
references = referencesInput --trainInput:index(1, torch.LongTensor({1,2,3,4,5}))
--print(references)

