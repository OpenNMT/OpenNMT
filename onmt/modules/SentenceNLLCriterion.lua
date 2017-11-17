--[[
  Define SentenceNLLCriterion.
  Implements Sentence-level log-likelihood as described in
  Collobert et al., Natural Language Processing (almost) from Scratch, JMLR 12(2011).
  
  This class tries to be both nn.Criterion and nn.Module at the same time.
  (Criterion with learnable parameters that are required for run-time.)
  
  This module requires double-precision calculations so internally, input/model parameters/output are cloned as double
  then converted back to moddel default types after the calculations.   
--]]
local SentenceNLLCriterion, parent = torch.class('onmt.SentenceNLLCriterion', 'nn.Criterion')

function SentenceNLLCriterion:__init(args, outputSize)
  parent.__init(self)

  if torch.type(outputSize) == 'table' then
    outputSize = outputSize[1]
  end

  local N = outputSize
  self.outputSize = N

  self.A0 = torch.zeros(N)        -- TagSize (N)
  self.A = torch.zeros(N, N)      -- TagSize (N) x TagSize (N)
  self.dA0 = torch.zeros(N)
  self.dA = torch.zeros(N,N)

  if args.max_grad_norm then
    self.max_grad_norm = args.max_grad_norm
  else
    self.max_grad_norm = 5
  end
end

function SentenceNLLCriterion:initCache()
  local N = self.outputSize

  self.dtype = onmt.utils.Cuda.activated and 'torch.CudaDoubleTensor' or 'torch.DoubleTensor'

  self.cache_dA_sum = torch.Tensor():type(self.dtype):resize(N,N)
  self.cache_dA0_sum = torch.Tensor():type(self.dtype):resize(N)
  self.cache_dA_tmp = torch.Tensor():type(self.dtype):resize(N,N)
  self.cache_dA0_tmp = torch.Tensor():type(self.dtype):resize(N)
  self.cache_path_transition_probs = torch.Tensor():type(self.dtype):resize(N,N)

  self.cache_F = torch.Tensor():type(self.dtype):resize(1,N,1)
  self.cache_dF = torch.Tensor():type(self.dtype):resize(1,N,1)
  self.cache_delta = torch.Tensor():type(self.dtype):resize(1,N,1)
  self.cache_delta_tmp = torch.Tensor():type(self.dtype):resize(N,N)
end

function SentenceNLLCriterion:training()
  self:initCache()
end

function SentenceNLLCriterion:evaluate()
  self:renormalizeParams()
end

function SentenceNLLCriterion:release()
  self.dtype = nil
  self.cache_dA_sum = nil
  self.cache_dA0_sum = nil
  self.cache_dA_tmp = nil
  self.cache_dA0_tmp = nil
  self.cache_path_transition_probs = nil
  self.cache_F = nil
  self.cache_dF = nil
  self.cache_delta = nil
  self.cache_delta_tmp = nil

  self.dA0 = nil
  self.dA = nil
end

function SentenceNLLCriterion:float()
  if self.A0 then self.A0 = self.A0:float() end
  if self.A then self.A = self.A:float() end
  if self.dA0 then self.dA0 = self.dA0:float() end
  if self.dA then self.dA = self.dA:float() end
end

function SentenceNLLCriterion:clearState()
end

function SentenceNLLCriterion:normalizeParams()
  local N = self.outputSize

  self.A0:add(-self.A0:min() + 0.000001)
  self.A0:div(self.A0:sum())
  self.A0:log()
  self.A:add(-torch.min(self.A,2):expand(N, N) + 0.000001)
  self.A:cdiv(self.A:sum(2):expand(N, N))
  self.A:log()
end

function SentenceNLLCriterion:renormalizeParams()
  self.A0:exp()
  self.A:exp()
  self:normalizeParams()
end

function SentenceNLLCriterion:postParametersInitialization()
  self:normalizeParams()
end

function SentenceNLLCriterion:parameters()
  return {self.A0, self.A}, {self.dA0, self.dA}
end

--[[
Copied from nn.Module
--]]
function SentenceNLLCriterion:getParameters()
  -- get parameters
  local parameters,gradParameters = self:parameters()
  local p, g = nn.Module.flatten(parameters), nn.Module.flatten(gradParameters)
  assert(p:nElement() == g:nElement(),
    'check that you are sharing parameters and gradParameters')
  if parameters then
    for i=1,#parameters do
      assert(parameters[i]:storageOffset() == gradParameters[i]:storageOffset(),
        'misaligned parameter at ' .. tostring(i))
    end
  end
  return p, g
end

--[[
Copied from nn.Module
--]]
function SentenceNLLCriterion:apply(callback)
  callback(self)
  if self.modules then
    for _, module in ipairs(self.modules) do
      module:apply(callback)
    end
  end
end

--[[
  Viterbi search
--]]
function SentenceNLLCriterion:viterbiSearch(input, sourceSizes)
  -- Input
  --   input:       BatchSize (B) x TagSize (N) x SeqLen (T): log-scale emission probabilities
  --   sourceSizes: BatchSize (B) of data type Long
  -- Output
  --   preds:       BatchSize (B) x SeqLen (T) of data type Long (index)

  local F = input
  local B = input:size(1)
  local N = input:size(2) -- should equal self.outputSize
  local T = input:size(3)

  local preds = onmt.utils.Cuda.convert(torch.LongTensor(B,T+1):zero()) -- extra dimension in T for EOS handling

  function _viterbiSearch_batch()
    -- OpenNMT mini batches are currently padded to the left of source sequences
    local isOnMask = onmt.utils.Cuda.convert(torch.ones(B, T+1))
    local isA0Mask = onmt.utils.Cuda.convert(torch.zeros(B, T+1))
    for b = 1, B do
      for t = 1, (T - sourceSizes[b]) do
        isOnMask[{b,t}] = 0
      end
      isA0Mask[{b, T+1-sourceSizes[b]}] = 1
    end
    local isAMask = isOnMask - isA0Mask
    local isMaxMask = isAMask
    local isFMask = isOnMask[{{}, {1,T}}]

    local maxScore = onmt.utils.Cuda.convert(torch.Tensor(B, N, T+1))
    local backPointer = onmt.utils.Cuda.convert(torch.LongTensor(B, N, T+1))

    -- A0
    local A0Score = nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(self.A0, 1):expand(N, N), 1):expand(B, N, N)
    -- A
    local AScore = nn.utils.addSingletonDimension(self.A:t(), 1):expand(B, N, N)

    for t = 1, T + 1 do
      local scores = onmt.utils.Cuda.convert(torch.Tensor(B, N, N):zero())

      local A0ScoreMasked = torch.cmul(A0Score, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isA0Mask[{{},t}],2),3):expand(B, N, N))
      scores:add(A0ScoreMasked)

      if t > 1 then
        local AScoreMasked = torch.cmul(AScore, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isAMask[{{},t}],2),3):expand(B, N, N))
        scores:add(AScoreMasked)

        -- maxScore
        local MaxScore = nn.utils.addSingletonDimension(maxScore[{{},{},t-1}],2):expand(B, N, N)
        local MaxScoreMasked = torch.cmul(MaxScore, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isMaxMask[{{},t}],2),3):expand(B, N, N))
        scores:add(MaxScoreMasked)
      end

      if t < T + 1 then
        -- F
        local FScore = nn.utils.addSingletonDimension(F[{{},{},t}],3):expand(B, N, N)
        local FScoreMasked = torch.cmul(FScore, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isFMask[{{},t}],2),3):expand(B, N, N))
        scores:add(FScoreMasked)
      end

      maxScore[{{},{},t}], backPointer[{{},{},t}] = scores:max(3)
    end

    for b=1,B do
      local pred = preds[b]
      _, pred[T+1] = maxScore[{b,{},T+1}]:max(1)
      for t=T+1,2+(T-sourceSizes[b]),-1 do
        pred[t-1] = backPointer[{b,pred[t],t}]
      end
    end
  end

  function _viterbiSearch_loop()
    local maxScore = onmt.utils.Cuda.convert(torch.Tensor(N, T+1))
    local backPointer = onmt.utils.Cuda.convert(torch.LongTensor(N, T+1))

    for b = 1, B do
      maxScore:zero()
      backPointer:zero()

      -- OpenNMT mini batches are currently padded to the left of source sequences
      local tOffset = T - sourceSizes[b]

      maxScore[{{},1+tOffset}], backPointer[{{},1+tOffset}] = nn.utils.addSingletonDimension(torch.add(self.A0, F[{b,{},1+tOffset}]),2):max(2)

      for t = 2+tOffset, T+1 do
        local scores = torch.add(nn.utils.addSingletonDimension(maxScore[{{},t-1}],1):expand(N, N), self.A:t())
        if t <= T then
          scores:add(nn.utils.addSingletonDimension(F[{b,{},t}],2):expand(N,N))
        end
        maxScore[{{},t}], backPointer[{{},t}] = scores:max(2)
      end

      local pred = preds[b]
      _, pred[T+1] = maxScore[{{},T+1}]:max(1)
      for t=T+1,2+tOffset,-1 do
        pred[t-1] = backPointer[{pred[t], t}]
      end
    end
  end

  _viterbiSearch_batch()
--  _viterbiSearch_loop()

  return preds[{{},{1,T}}]:clone()
end

function logsumexp(x)
  -- Input
  --   x:  TagSize (N) or TagSize (N) x TagSize (N)

  local N = x:size(1) -- should equal self.outputSize

  local max, _ = x:max(1)  --  1 or 1 x N
  local log_sum_exp = (x - torch.repeatTensor(max, N, 1)):exp():sum(1):log() -- 1 or 1 x N
  -- find NaN values and assign a valid value
  local NaN_mask = log_sum_exp:ne(log_sum_exp)
  log_sum_exp[NaN_mask] = max:max()
  return log_sum_exp:add(max):squeeze(1) -- 1 or N
end

function SentenceNLLCriterion:updateOutput(input, target)

  -- Input variables
  --   input: BatchSize (B) x TagSize (N) x SeqLen (T)
  --   target: BatchSize (B) x SeqLen (T)

  -- Output variable
  --   loss

  local Y = target
  local B = input:size(1)
  local N = input:size(2) -- should equal self.outputSize
  local T = input:size(3) + 1 -- extra T dimension for EOS

  if not self.cache_F then -- Initialize cache when updateOutput() is called at inference time (e.g. to compute loss)
    self:initCache()
  end

  local F = self.cache_F:resize(B, N, T)
  F[{{},{},{1,input:size(3)}}]:copy(input)
  F[{{},                 {}, -1}] = 0.000001
  F[{{}, onmt.Constants.EOS, -1}] = 1
  F[{{},                 {}, -1}]:log()

  self.cache_delta:resize(F:size()):zero() -- B,N,T

  local A0_dtype = self.A0:type(self.dtype)
  local A_dtype = self.A:type(self.dtype)

  local loss = 0.0

  -- TODO vectorize for Batch dimension
  for b = 1, B do

    local delta = self.cache_delta[b]

    -- OpenNMT mini batches are currently padded to the left of source sequences
    local tOffset = 0
    while Y[{b,1+tOffset}] == onmt.Constants.PAD do
      tOffset = tOffset + 1
    end

    -- init state
    local t = 1 + tOffset
    local referenceScore = self.A0[Y[b][t]] + F[b][Y[b][t]][t]
    delta[{{},t}]:add(F[{b,{},t}], A0_dtype)

    -- fwd transition recursion
    for t = 2 + tOffset, T do
      local Y_t = Y[b][t]
      local Y_t_1 = Y[b][t-1]

      referenceScore = referenceScore + self.A[Y_t_1][Y_t] + F[b][Y_t][t]

      self.cache_delta_tmp:add(A_dtype, nn.utils.addSingletonDimension(delta[{{},t-1}],2):expand(N,N))
      delta[{{},t}]:add(F[{b,{},t}], logsumexp(self.cache_delta_tmp))
    end

    local loglik = referenceScore - logsumexp(delta[{{},T}])
    loss = loss - loglik
  end

  return loss
end

function SentenceNLLCriterion:updateGradInput(input, target)
  -- Input: F, A0, A
  -- Output: dF, dA0, dA w.r.t Loss in target

  local dF = self.cache_dF:resize(input:size()):zero()
  local Y = target
  local B = input:size(1)
  local N = input:size(2) -- should equal self.outputSize
  local T = input:size(3)+1

  --  print('A0:\n' .. tostring(self.A0))
  --  print('A:\n' .. tostring(self.A))
  --  print('Y:\n' .. tostring(Y))
  --  print('input:\n' .. tostring(input))
  --  print('F:\n' .. tostring(F))

  local A_dtype = self.A:type(self.dtype)

  local dA_sum = self.cache_dA_sum:zero()
  local dA0_sum = self.cache_dA0_sum:zero()

  -- TODO vectorize for Batch dimension
  for b = 1, B do
    -- OpenNMT mini batches are currently padded to the left of source sequences
    local tOffset = 0
    while Y[{b,1+tOffset}] == onmt.Constants.PAD do
      tOffset = tOffset + 1
    end

    local delta = self.cache_delta[b]    --  N x T

    local dA = self.cache_dA_tmp:zero()
    local dA0 = self.cache_dA0_tmp:zero()

    local deriv_Clogadd = delta[{{},T}]:exp()
    deriv_Clogadd:div(deriv_Clogadd:sum())
    deriv_Clogadd[deriv_Clogadd:ne(deriv_Clogadd)] = 0

    for t= T, (2+tOffset), -1 do

      local Y_t = Y[b][t]
      local Y_t_1 = Y[b][t-1]

      if t < T then -- dF for the last EOS token does not exist
        dF[{b,Y_t,t}] = dF[{b,Y_t,t}] - 1
      end
      dA[{Y_t_1,Y_t}] = dA[{Y_t_1,Y_t}] - 1

      -- compute and add partial derivatives w.r.t transition scores
      local path_transition_probs = self.cache_path_transition_probs
      path_transition_probs:add(A_dtype, nn.utils.addSingletonDimension(delta[{{},t-1}],2):expand(N,N))
      path_transition_probs:exp()
      path_transition_probs:cdiv(path_transition_probs:sum(1):expand(N,N))
      path_transition_probs[path_transition_probs:ne(path_transition_probs)] = 0

      if t < T then
        dF[{b,{},t}]:add(deriv_Clogadd)
      end
      path_transition_probs:cmul(nn.utils.addSingletonDimension(deriv_Clogadd, 1):expand(N,N))
      local dAt = path_transition_probs
      dA:add(dAt)
      deriv_Clogadd = dAt:sum(2):squeeze(2)

      onmt.train.Optim.clipGradByNorm({deriv_Clogadd}, self.max_grad_norm)
    end

    local t = 1 + tOffset
    local Y_t = Y[b][t]
    dF[{b,Y_t,t}] = dF[{b,Y_t,t}] - 1
    dA0[Y_t] = dA0[Y_t] - 1

    dF[{b,{},t}]:add(deriv_Clogadd)
    dA0:add(deriv_Clogadd)

    dA_sum:add(dA)
    dA0_sum:add(dA0)
  end

  --  print('dA0:\n' .. tostring(dA0_sum))
  --  print('dA:\n' .. tostring(dA_sum))
  --  print('dF:\n' .. tostring(dF))

  self.dA:add(onmt.utils.Cuda.convert(dA_sum/B))
  self.dA0:add(onmt.utils.Cuda.convert(dA0_sum/B))

  if not self.gradInput then
    self.gradInput = onmt.utils.Cuda.convert(torch.Tensor())
  end
  self.gradInput:resize(input:size())
  self.gradInput:copy(dF)

  return self.gradInput
end

--function SentenceNLLCriterion:updateParameters(learningRate)
--  self.A0:add(-learningRate * self.dA0)
--  self.A:add(-learningRate * self.dA)
--end
--
--function SentenceNLLCriterion:zeroGradParameters()
--  self.dA0:zero()
--  self.dA:zero()
--end

--[[ Return a new SentenceNLLCriterion using the serialized data `pretrained`. ]]
function SentenceNLLCriterion.load(pretrained)
  local self = torch.factory('onmt.SentenceNLLCriterion')()

  parent.__init(self)
  self.A0 = pretrained.A0
  self.A = pretrained.A
  self.outputSize = pretrained.outputSize
  self.max_grad_norm = pretrained.max_grad_norm

  return self
end

--[[ Return data to serialize. ]]
function SentenceNLLCriterion:serialize()
  return {
    A0 = self.A0,
    A = self.A,
    outputSize = self.outputSize,
    max_grad_norm = self.max_grad_norm,
    float = self.float,
    clearState = self.clearState,
    apply = self.apply
  }
end
