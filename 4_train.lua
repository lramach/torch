----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

print '==> defining some tools'


-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(#classes, classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
validationLogger = optim.Logger(paths.concat(opt.save, 'validation.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

print(optimMethod)

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local charFeats = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
	 -- print(shuffle[i])
         local input = trainData.data[shuffle[i]]
	 --extracting char grams
	 local charGrams = charGramsInputTrain[shuffle[i]]:double()
	 -- target has to be indices of the classes
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
         --print(input)
	 --print(charGrams)
	 table.insert(inputs, input)
         table.insert(charFeats, charGrams)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
		       n = #inputs
                       for i = 1,#inputs do
                          -- estimate f
			  local input = inputs[i]
			  local charGrams = charFeats[i]
			  -- print(i)
			  input = input[input:ne(-1)]
			  if input:nElement() == 0 then
			    n = n-1
			  else
  			    weOut = weSum:forward(input)
  			    input = torch.Tensor(charGrams:size(1)+weOut:size(1))
  			    input[{{1,charGrams:size(1)}}] = charGrams
  			    input[{{charGrams:size(1)+1, input:size(1)}}] = weOut 
  			    local output = model:forward(input)
  			    local err = criterion:forward(output,targets[i]) 
  			    f = f + err
  
                            -- estimate df/dW
                            local df_do = criterion:backward(output, targets[i])
                            --print(df_do)
  			    model:backward(input, df_do)
  
                            -- update confusion
                            confusion:add(output, targets[i])
			  end
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(n)
                       f = f/n

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model' .. opt.prompt .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
