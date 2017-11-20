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

function SentenceNLLCriterion:training()
  self:_initTrainCache()
end

function SentenceNLLCriterion:evaluate()
  self:renormalizeParams()
end

function SentenceNLLCriterion:release()
  self:_freeTrainCache()
  self:_freeViterbiCache()
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

  if not self.cache_viterbi_preds then
    self:_initViterbiCache()
  end

  local preds = self.cache_viterbi_preds:resize(B,T+1):zero() -- extra dimension in T for EOS handling

  function _viterbiSearch_batch()
    -- OpenNMT mini batches are padded to the left of source sequences
    local isOnMask = self.cache_viterbi_isOnMask:resize(B,T+1):fill(1)
    local isA0Mask = self.cache_viterbi_isA0Mask:resize(B,T+1):zero()
    for b = 1, B do
      for t = 1, (T - sourceSizes[b]) do
        isOnMask[{b,t}] = 0
      end
      isA0Mask[{b, T+1-sourceSizes[b]}] = 1
    end
    local isAMask = self.cache_viterbi_isAMask:add(isOnMask, -isA0Mask)
    local isMaxMask = isAMask
    local isFMask = isOnMask[{{}, {1,T}}]

    local maxScore = self.cache_viterbi_maxScore:resize(B, N, T+1)
    local backPointer = self.cache_viterbi_backPointer:resize(B, N, T+1)

    -- A0
    local A0Score = nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(self.A0, 1):expand(N, N), 1):expand(B, N, N)
    -- A
    local AScore = nn.utils.addSingletonDimension(self.A:t(), 1):expand(B, N, N)

    for t = 1, T + 1 do
      local scores = self.cache_viterbi_scores:resize(B,N,N):zero()

      local A0ScoreMasked = self.cache_viterbi_XScoreMasked:resize(B,N,N)
      A0ScoreMasked:cmul(A0Score, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isA0Mask[{{},t}],2),3):expand(B, N, N))
      scores:add(A0ScoreMasked)

      if t > 1 then
        local AScoreMasked = self.cache_viterbi_XScoreMasked:resize(B,N,N)
        AScoreMasked:cmul(AScore, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isAMask[{{},t}],2),3):expand(B, N, N))
        scores:add(AScoreMasked)

        -- maxScore
        local MaxScore = nn.utils.addSingletonDimension(maxScore[{{},{},t-1}],2):expand(B, N, N)
        local MaxScoreMasked = self.cache_viterbi_XScoreMasked:resize(B,N,N)
        MaxScoreMasked:cmul(MaxScore, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isMaxMask[{{},t}],2),3):expand(B, N, N))
        scores:add(MaxScoreMasked)
      end

      if t < T + 1 then
        -- F
        local FScore = nn.utils.addSingletonDimension(F[{{},{},t}],3):expand(B, N, N)
        local FScoreMasked = self.cache_viterbi_XScoreMasked:resize(B,N,N)
        FScoreMasked:cmul(FScore, nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isFMask[{{},t}],2),3):expand(B, N, N))
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

      -- OpenNMT mini batches are padded to the left of source sequences
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
  local log_sum_exp
  if x:nDimension() == 1 then
    log_sum_exp = (x - max:expand(N)):exp():sum(1):log() -- 1
  else
    log_sum_exp = (x - max:expand(N,N)):exp():sum(1):log() -- 1 x N
  end
  -- find NaN values and assign a valid value
  local NaN_mask = log_sum_exp:ne(log_sum_exp)
  log_sum_exp[NaN_mask] = max:max()
  return log_sum_exp:add(max):squeeze(1) -- 1 or N
end

function logsumexp_batch(x)
  -- Input
  --   x:  B x N or B x N x N

  local B = x:size(1)
  local N = x:size(2) -- should equal self.outputSize

  local max, _ = x:max(2)  --  B x 1  or B x 1 x N
  local log_sum_exp
  if x:nDimension() == 2 then
    log_sum_exp = (x - max:expand(B,N)):exp():sum(2):log() -- B x 1
  else
    log_sum_exp = (x - max:expand(B,N,N)):exp():sum(2):log() -- B x 1 x N
  end
  -- find NaN values and assign a valid value
  local NaN_mask = log_sum_exp:ne(log_sum_exp)
  log_sum_exp[NaN_mask] = max:max()
  return log_sum_exp:add(max):squeeze(2) -- B x 1 or B x N
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
    self:_initTrainCache()
  end

  local F = self.cache_F:resize(B, N, T)
  F[{{},{},{1,input:size(3)}}]:copy(input)
  F[{{},                 {}, -1}] = 0.000001
  F[{{}, onmt.Constants.EOS, -1}] = 1
  F[{{},                 {}, -1}]:log()

  self.cache_delta:resize(F:size()):zero() -- B,N,T

  self.cache_A0_dtype = self.A0:type(self.dtype)
  self.cache_A_dtype = self.A:type(self.dtype)

  local loss = 0.0

  function _updateOutput_batch()
    -- OpenNMT mini batches are padded to the left of source sequences
    local isOnMask = self.cache_isOnMask:resize(B,T):fill(1)
    local isA0Mask = self.cache_isA0Mask:resize(B,T):zero()
    local yLookupTensor = self.cache_dF:resize(B,N,T):zero()
    for b = 1, B do
      for t = 1, T do
        if Y[b][t] == onmt.Constants.PAD then
          isOnMask[b][t] = 0
        else
          if t == 1 or (t > 1 and Y[b][t-1] == onmt.Constants.PAD) then
            isA0Mask[b][t] = 1
          end
          yLookupTensor[b][ Y[b][t] ][t] = 1
        end
      end
    end
    local isAMask = self.cache_isAMask:add(isOnMask, -isA0Mask)
    local isFMask = isOnMask

    local delta = self.cache_delta
    local delta_tmp = self.cache_delta_tmp:resize(B,N,N)

    local refScores = torch.Tensor(B):type(self.dtype):zero()
    local logLiks = torch.Tensor(B):type(self.dtype):zero()

    -- A0
    self.cache_A0_dtype_batch = nn.utils.addSingletonDimension(self.cache_A0_dtype, 1):expand(B,N)
    local A0_dtype_batch = self.cache_A0_dtype_batch
    -- A
    self.cache_A_dtype_batch = nn.utils.addSingletonDimension(self.cache_A_dtype, 1):expand(B,N,N)
    local A_dtype_batch = self.cache_A_dtype_batch

    for t = 1, T do
      -- refScore
      refScores:add(torch.cmul(A0_dtype_batch,
                              yLookupTensor[{{},{},t}])
                        :cmul(nn.utils.addSingletonDimension(isA0Mask[{{},t}],2):expand(B,N))
                     :sum(2):squeeze(2) -- B
                  )
      refScores:add(torch.cmul(F[{{},{},t}], yLookupTensor[{{},{},t}]):sum(2):squeeze(2)) -- B; yLookupTensor is already masked
      if t > 1 then
        refScores:add(torch.cmul(A_dtype_batch,
                        nn.utils.addSingletonDimension(yLookupTensor[{{},{},t-1}],3):expand(B,N,N)):sum(2) -- select t-1
                        :cmul(yLookupTensor[{{},{},t}]):sum(3):squeeze(3):squeeze(2) -- select t
                 ) -- B; yLookupTensor is already masked
      end

      --loglik
      delta[{{},{},t}]:add(torch.cmul(A0_dtype_batch, nn.utils.addSingletonDimension(isA0Mask[{{},t}],2):expand(B,N))) -- B x N
      delta[{{},{},t}]:add(torch.cmul(F[{{},{},t}], nn.utils.addSingletonDimension(isFMask[{{},t}],2):expand(B,N)))
      if t > 1 then
        delta_tmp:add( A_dtype_batch,
                       nn.utils.addSingletonDimension(delta[{{},{},t-1}],3):expand(B,N,N)
                     ):cmul(nn.utils.addSingletonDimension(nn.utils.addSingletonDimension(isAMask[{{},t}],2),3):expand(B,N,N)) -- B x N x N

        delta[{{},{},t}]:add(logsumexp_batch(delta_tmp)) -- B x N
      end
    end

    logLiks:add(refScores, -logsumexp_batch(delta[{{},{},T}]))
    loss = -logLiks:sum()
  end

  function _updateOutput_loop()
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
      delta[{{},t}]:add(self.cache_A0_dtype, F[{b,{},t}])

      -- fwd transition recursion
      for t = 2 + tOffset, T do
        local Y_t = Y[b][t]
        local Y_t_1 = Y[b][t-1]

        referenceScore = referenceScore + self.A[Y_t_1][Y_t] + F[b][Y_t][t]

        self.cache_delta_tmp:add(self.cache_A_dtype, nn.utils.addSingletonDimension(delta[{{},t-1}],2):expand(N,N))
        delta[{{},t}]:add(F[{b,{},t}], logsumexp(self.cache_delta_tmp))
      end

      local loglik = referenceScore - logsumexp(delta[{{},T}])
      loss = loss - loglik
    end
  end

  _updateOutput_batch()
--  _updateOutput_loop()

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

  local dA_sum = self.cache_dA_sum:zero()
  local dA0_sum = self.cache_dA0_sum:zero()

  function _updateGradInput_batch()

    local delta = self.cache_delta    --  B x N x T; cached calculations from fwd path

    -- Assume cached masks and batched A0, A are still valid
    local isA0Mask = self.cache_isA0Mask
    local isFMask = self.cache_isOnMask

    local A_dtype_batch = self.cache_A_dtype_batch
    local path_transition_probs = self.cache_path_transition_probs:resize(B,N,N)
    local dA = self.cache_dA_tmp:resize(B,N,N):zero() -- B x N x N
    local dA0 = self.cache_dA0_tmp:resize(B,N):zero() -- B x N

    for b = 1, B do
      for t = 1, T do
        if Y[b][t] ~= onmt.Constants.PAD then
          if t == 1 or (t > 1 and Y[b][t-1] == onmt.Constants.PAD) then
            dA0[{b,Y[b][t]}] = dA0[{b,Y[b][t]}] - 1 -- A0
          end
          if t > 1 then
            dA[{b,Y[b][t-1],Y[b][t]}] = dA[{b,Y[b][t-1],Y[b][t]}] - 1 -- A
          end
          if t < T then -- dF for the last EOS token does not exist
            dF[{b,Y[b][t],t}] = dF[{b,Y[b][t],t}] - 1 -- F
          end
        end
      end
    end

    local deriv_Clogadd = delta[{{},{},T}]:exp()
    deriv_Clogadd:cdiv(deriv_Clogadd:sum(2):expand(B,N))
    deriv_Clogadd[deriv_Clogadd:ne(deriv_Clogadd)] = 0

    for t = T,1,-1 do
      if t < T then
        -- F
        dF[{{},{},t}]:add(torch.cmul(deriv_Clogadd, nn.utils.addSingletonDimension(isFMask[{{},t}],2):expand(B,N)))
        -- A0
        dA0:add(torch.cmul(deriv_Clogadd, nn.utils.addSingletonDimension(isA0Mask[{{},t}],2):expand(B,N)))
      end
      -- A
      if t > 1 then
        path_transition_probs:add(A_dtype_batch,
                                  nn.utils.addSingletonDimension(
                                     delta[{{},{},t-1}] -- delta is calculated from masked
                                     ,3):expand(B,N,N))
        path_transition_probs:exp()
        path_transition_probs:cdiv(path_transition_probs:sum(2):expand(B,N,N))
        path_transition_probs[path_transition_probs:ne(path_transition_probs)] = 0
        path_transition_probs:cmul(nn.utils.addSingletonDimension(deriv_Clogadd, 2):expand(B,N,N))

        local dAt = path_transition_probs
        dA:add(dAt)

        deriv_Clogadd = dAt:sum(3):squeeze(3)
        for b = 1, B do
          onmt.train.Optim.clipGradByNorm({deriv_Clogadd[b]}, self.max_grad_norm)
        end
      end
    end

    dA_sum:add(dA:sum(1):squeeze(1))
    dA0_sum:add(dA0:sum(1):squeeze(1))
  end

  function _updateGradInput_loop()
    local A_dtype = self.cache_A_dtype

    for b = 1, B do
      -- OpenNMT mini batches are padded to the left of source sequences
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
  end

--  _updateGradInput_loop()
  _updateGradInput_batch()

  self.dA:add(onmt.utils.Cuda.convert(dA_sum/B))
  self.dA0:add(onmt.utils.Cuda.convert(dA0_sum/B))

  if not self.gradInput then
    self.gradInput = onmt.utils.Cuda.convert(torch.Tensor())
  end
  self.gradInput:resize(input:size())
  self.gradInput:copy(dF)

  return self.gradInput
end

function SentenceNLLCriterion:_initViterbiCache()
  local N = self.outputSize
  self.cache_viterbi_preds = onmt.utils.Cuda.convert(torch.LongTensor(1,1))
  self.cache_viterbi_maxScore = onmt.utils.Cuda.convert(torch.Tensor(1, N, 1))
  self.cache_viterbi_backPointer = onmt.utils.Cuda.convert(torch.LongTensor(1, N, 1))
  self.cache_viterbi_isOnMask = onmt.utils.Cuda.convert(torch.Tensor(1, 1))
  self.cache_viterbi_isA0Mask = onmt.utils.Cuda.convert(torch.Tensor(1, 1))
  self.cache_viterbi_isAMask = onmt.utils.Cuda.convert(torch.Tensor(1, 1))
  self.cache_viterbi_scores = onmt.utils.Cuda.convert(torch.Tensor(1, N, N))
  self.cache_viterbi_XScoreMasked = onmt.utils.Cuda.convert(torch.Tensor(1, N, N))
end

function SentenceNLLCriterion:_freeViterbiCache()
  self.cache_viterbi_preds = nil
  self.cache_viterbi_maxScore = nil
  self.cache_viterbi_backPointer = nil
  self.cache_viterbi_isOnMask = nil
  self.cache_viterbi_isA0Mask = nil
  self.cache_viterbi_isAMask = nil
  self.cache_viterbi_scores = nil
  self.cache_viterbi_XScoreMasked = nil
end

function SentenceNLLCriterion:_initTrainCache()
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

  self.cache_isOnMask = torch.Tensor(1, 1):type(self.dtype)
  self.cache_isA0Mask = torch.Tensor(1, 1):type(self.dtype)
  self.cache_isAMask = torch.Tensor(1, 1):type(self.dtype)
end

function SentenceNLLCriterion:_freeTrainCache()
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

  self.cache_isOnMask = nil
  self.cache_isA0Mask = nil
  self.cache_isAMask = nil
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
