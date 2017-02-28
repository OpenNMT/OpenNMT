--[[ Data management and batch creation. Handles data created by `preprocess.lua`. ]]
require 'torchx'

local SampledDataset = torch.class("SampledDataset")

--[[ Initialize a data object given aligned tables of IntTensors `srcData`
  and `tgtData`.
--]]
function SampledDataset:__init(srcData, tgtData, samplingSize, sample_w_ppl, sample_w_ppl_init, sample_w_ppl_max)

  self.src = srcData.words
  self.srcFeatures = srcData.features

  if tgtData ~= nil then
    self.tgt = tgtData.words
    self.tgtFeatures = tgtData.features
  end

  assert(samplingSize <= #self.src, "sampling size should be less than the number of instances in the train data")

  self.samplingSize = samplingSize
  self.sample_w_ppl = sample_w_ppl
  self.sample_w_ppl_init = sample_w_ppl_init
  self.sample_w_ppl_max = sample_w_ppl_max
  self.startedPplSampling = false

  if self.sample_w_ppl then
    self.samplingProb = torch.Tensor(#self.src)
    self.samplingProb:fill(self.sample_w_ppl_init)
  else
    self.samplingProb = torch.ones(#self.src)
  end

  self.ppl = torch.Tensor(#self.src)
  self.ppl:fill(self.sample_w_ppl_init)
  self.isSampled = torch.zeros(#self.src)
end

--[[ initiate sampling ]]
function SampledDataset:sample(avgPpl)
  _G.logger:info('Sampling...')

  -- populate self.samplingProb with self.ppl if average ppl is below self.sample_w_ppl_init
  if avgPpl == nil and self.sample_w_ppl then
    avgPpl = torch.sum(self.ppl)
    avgPpl = avgPpl/self.ppl:size(1)
  end

  if self.sample_w_ppl and not self.startedPplSampling and avgPpl < self.sample_w_ppl_init then
    _G.logger:info('Beginning to sample with ppl as probability distribution...')
    self.startedPplSampling = true
  end

  if self.startedPplSampling then

    local threshold = self.sample_w_ppl_max

    if self.sample_w_ppl_max == -1 then
      -- use mode (instead of mean) and stdev of samples with ppl>=mode to
      -- find max ppl to consider (mode + x * stdev). when x is:
      --      x: 1 ~ 100% - 31.7%/2 of train data are covered (divide by 2 because we cut only one-tail)
      --      x: 2 ~ 100% - 4.55%/2
      --      x: 3 ~ 100% - 0.270%/2
      --      x: 4 ~ 100% - 0.00633%/2
      --      x: 5 ~ 100% - 0.0000573%/2
      --      x: 6 ~ 100% - 0.000000197%/2
      --  (https://en.wikipedia.org/wiki/Standard_deviation)
      -- we are using mode instead of average, and only samples above mode to calculate stdev, so
      -- this is not really theoretically valid numbers, but more for emperical uses

      -- find mode
      local pplRounded = torch.round(self.ppl)
      local bin = {}
      for i = 1, pplRounded:size(1) do
        local idx = pplRounded[i]
        if bin[idx] == nil then
          bin[idx] = 0
        end
        bin[idx] = bin[idx] + 1
      end
      local mode = nil
      for key,value in pairs(bin) do
        if mode == nil or bin[mode] < value then
          mode = key
        end
      end

      -- stdev with mode only using samples with ppl >= mode
      local sum = 0
      local cnt = 0
      for i = 1, self.ppl:size(1) do
        if self.ppl[i] > mode then
          sum = math.pow(self.ppl[i]-mode, 2)
          cnt = cnt + 1
        end
      end
      local stdev = math.sqrt(sum/(cnt-1))

      threshold = mode + stdev

      _G.logger:info('Sampler mode: ' .. mode)
      _G.logger:info('Sampler stdev: ' .. stdev)
      _G.logger:info('Sampler threshold: ' .. threshold)
    end

    for i = 1, self.ppl:size(1) do
      if self.ppl[i] ~= self.sample_w_ppl_init and self.ppl[i] > threshold then
        -- asign low value to instances with ppl above threshold (outliers)
        self.samplingProb[i] = 1
      else
        self.samplingProb[i] = self.ppl[i]
      end
    end
  end

--  local sampled = torch.multinomial(self.samplingProb:view(1,#self.src), self.samplingSize, --[[replacement]] false)
-- Faster sampling
-- https://github.com/nicholas-leonard/torchx/blob/master/AliasMultinomial.lua
  local sampler = torch.AliasMultinomial(self.samplingProb)
  _G.logger:info('Created sampler...')
  local sampled = torch.LongTensor(self.samplingSize)
  sampled = sampler:batchdraw(sampled)
  _G.logger:info('Sampled...')

  self.isSampled:zero()
  for i=1, sampled:size(1) do
    self.isSampled[sampled[i]] = 1
  end

  -- Prepares batches in terms of range within self.src and self.tgt.
  self.batchRange = {}
  local offset = 0
  local batchSize = 1
  local sourceLength = 0

  for i = 1, #self.src do
    if self.isSampled[i] > 0 then
      -- Set up the offsets to make same source size batches of the
      -- correct size.
      if batchSize == self.maxBatchSize or self.src[i]:size(1) ~= sourceLength then
        if offset > 1 then
          table.insert(self.batchRange, { ["begin"] = offset, ["end"] = i - 1 })
--          print('Batch ' .. #self.batchRange .. ' (' .. offset .. ', ' .. (i-1) .. '): ' .. batchSize)
        end

        offset = i
        batchSize = 1
        sourceLength = self.src[i]:size(1)
        targetLength = 0
      else
        batchSize = batchSize + 1
      end
    end
  end
  -- Catch last batch.
  table.insert(self.batchRange, { ["begin"] = offset, ["end"] = #self.src })
--  print('Batch ' .. #self.batchRange .. ' (' .. offset .. ', ' .. #self.src .. '): ' .. batchSize)
end

--[[ get ppl ]]
function SampledDataset:getPpl()
  return self.ppl
end

--[[ set ppl ]]
function SampledDataset:setPpl(batchIdx, ppl)

  -- assert batchIdx <= #self.batchRange
  local rangeStart = self.batchRange[batchIdx]["begin"]
  local rangeEnd = self.batchRange[batchIdx]["end"]

  local pplIdx = 1
  for i = rangeStart, rangeEnd do
    if self.isSampled[i] > 0 then
      self.ppl[i] = ppl[pplIdx]
      pplIdx = pplIdx + 1
    end
  end
end

--[[ Setup up the training data to respect `maxBatchSize`. ]]
function SampledDataset:setBatchSize(maxBatchSize)

  self.maxBatchSize = maxBatchSize
  self.maxSourceLength = 0
  self.maxTargetLength = 0

  for i = 1, #self.src do
    self.maxSourceLength = math.max(self.maxSourceLength, self.src[i]:size(1))
    if self.tgt ~= nil then
      -- Target contains <s> and </s>.
      local targetSeqLength = self.tgt[i]:size(1) - 1
      self.maxTargetLength = math.max(self.maxTargetLength, targetSeqLength)
    end
  end

  self:sample()
end

--[[ Return number of batches. ]]
function SampledDataset:batchCount()
  if self.batchRange == nil then
    if #self.src > 0 then
      return 1
    else
      return 0
    end
  end
  return #self.batchRange
end

--[[ Get `Batch` number `idx`. If nil make a batch of all the data. ]]
function SampledDataset:getBatch(idx)
  if #self.src == 0 then
    return nil
  end

  if idx == nil or self.batchRange == nil then
    return onmt.data.Batch.new(self.src, self.srcFeatures, self.tgt, self.tgtFeatures)
  end

  local rangeStart = self.batchRange[idx]["begin"]
  local rangeEnd = self.batchRange[idx]["end"]

  local src = {}
  local tgt

  if self.tgt ~= nil then
    tgt = {}
  end

  local srcFeatures = {}
  local tgtFeatures = {}

  for i = rangeStart, rangeEnd do
    if self.isSampled[i] > 0 then
      table.insert(src, self.src[i])

      if self.srcFeatures[i] then
        table.insert(srcFeatures, self.srcFeatures[i])
      end

      if self.tgt ~= nil then
        table.insert(tgt, self.tgt[i])
        if self.tgtFeatures[i] then
          table.insert(tgtFeatures, self.tgtFeatures[i])
        end
      end
    end
  end

  return onmt.data.Batch.new(src, srcFeatures, tgt, tgtFeatures)
end

return SampledDataset
