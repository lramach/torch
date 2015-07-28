----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining validation procedure'

-- test function
function validation()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,validationData:size() do
      -- disp progress
      xlua.progress(t, validationData:size())

      -- get new sample
      local input = validationData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = validationData.labels[t]

      -- test sample
      input = input[input:ne(-1)]
      if input:nElement() > 0 then
        m1out = m1:forward(input)
        m2out = m2:forward(input)
        input = torch.Tensor(1+m2out:size(1))
        input[{{1,1}}] = m1out
        input[{{2, input:size(1)}}] = m2out
        local pred = model:forward(input)
        confusion:add(pred, target)
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / validationData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   validationLogger:add{['% mean class accuracy (validation set)'] = confusion.totalValid * 100}
   if opt.plot then
      validationLogger:style{['% mean class accuracy (validation set)'] = '-'}
      validationLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
