require 'hdf5';
-- reading in the glove vectors (word-embeddings)
myFile = hdf5.open('/users/lakshmiramachandran/Documents/pearson-datasets/misc-torch-code/glovevec.COSC120160.h5', 'r')
df = myFile:read():all()
glovevec = df.glovevec

--Reading the test dataset
myFile = hdf5.open('/users/lakshmiramachandran/Documents/pearson-datasets/misc-torch-code/COSC120160_test.h5', 'r')
df = myFile:read():all()
testInput = df.index
testLabels = df.target + 1
print '==> loading dataset'

-- Load the test data.
testData = {
  data = testInput, 
  labels = testLabels,
  size = function() return testInput:size(1) end
}
tesize = (#testInput)[1]
