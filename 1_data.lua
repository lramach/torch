require 'hdf5';
----parse command line args
--if not opt then
--  print '==> processing options'
--  cmd = torch.CmdLine()
--  cmd:text()
--  cmd:text('SVHN Dataset Preprocessing')
--  cmd:text()
--  cmd:text('Options:')
--  cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
--  cmd:option('-visualize', true, 'visualize input data and weights during training')
--  cmd:text()
--  opt = cmd:parse(arg or {})
--end

-- reading in the glove vectors (word-embeddings)
myFile = hdf5.open('/users/lakshmiramachandran/Documents/pearson-datasets/misc-torch-code/glovevec.COSC120160.h5', 'r')
df = myFile:read():all()
glovevec = df.glovevec

-- reading the input, contains the indices of the words
myFile = hdf5.open('/users/lakshmiramachandran/Documents/pearson-datasets/misc-torch-code/COSC120160_train.h5', 'r')
df = myFile:read():all()
trainInput = df.index 
-- input[input:eq(0/0)] = -1
-- input = input:int()
--labels are indices of the classes
trainLabels = df.target +1 
--torch.cat(torch.Tensor(10):fill(1), torch.Tensor(10):fill(2)) 

--reading the tst dataset
myFile = hdf5.open('/users/lakshmiramachandran/Documents/pearson-datasets/misc-torch-code/COSC120160_test.h5', 'r')
df = myFile:read():all()
testInput = df.index
testLabels = df.target + 1
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.
print((#trainInput)[1])
trainData = {
  data = trainInput,
  labels = trainLabels, 
  size = function() return trainInput:size(1) end
}
--
--
-- Finally we load the test data.
testData = {
  data = testInput, 
  labels = testLabels,
  size = function() return testInput:size(1) end
}
trsize = (#trainInput)[1]
tesize = (#testInput)[1]
