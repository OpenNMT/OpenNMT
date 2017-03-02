--[[ Data management and batch creation. Handles data created by `preprocess.lua`. ]]
require 'torchx'

local SampledDataset = torch.class("SampledDataset")

local options = {
  {'-sample',              0, [[Number of instances to sample from train data in each epoch]]},
  {'-sample_w_ppl',    false, [[use ppl as probability distribution when sampling]]},
  {'-sample_w_ppl_init',  15, [[start perplexity-based sampling when average train perplexity per batch falls below this value]]},
  {'-sample_w_ppl_max', -1.5, [[when greater than 0, instances with perplexity above this value will be considered as noise and ignored;
                                when less than 0, mode + (-sample_w_ppl_max) * stdev will be used as threshold]]},
}

function SampledDataset.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'SampledDataset')
end

--[[ Initialize a data object given aligned tables of IntTensors `srcData`
  and `tgtData`.
--]]
function SampledDataset:__init(srcData, tgtData, opt)

  self.src = srcData.words or srcData.vectors
  self.srcFeatures = srcData.features

  if tgtData ~= nil then
    self.tgt = tgtData.words  or tgtData.vectors
    self.tgtFeatures = tgtData.features
  end

  self.samplingSize = opt.sample
  self.sample_w_ppl = opt.sample_w_ppl
  self.sample_w_ppl_init = opt.sample_w_ppl_init
  self.sample_w_ppl_max = opt.sample_w_ppl_max
  self.startedPplSampling = false

  _G.logger:info(' * sampling ' .. opt.sample .. ' instances from ' .. #self.src .. ' at each epoch')
  if opt.sample_w_ppl then
    _G.logger:info(' * using train data perplexity as probability distribution when sampling')
    _G.logger:info(' * sample_w_ppl_init: ' .. opt.sample_w_ppl_init)
    _G.logger:info(' * sample_w_ppl_max: ' .. opt.sample_w_ppl_max)
  end

  if self.sample_w_ppl then
    self.samplingProb = torch.Tensor(#self.src)
    self.samplingProb:fill(self.sample_w_ppl_init)
    self.ppl = torch.Tensor(#self.src)
    self.ppl:fill(self.sample_w_ppl_init)
  else
    self.samplingProb = torch.ones(#self.src)
  end

  self.sampled = nil
end

function SampledDataset:checkModel(model)
  if not model.returnIndividualLosses or model:returnIndividualLosses(true) == false then
    _G.logger:info('Current model does not support training with invididual losses; Sampling with individual loss will be disabled.')
    self.sample_w_ppl = false
    self.samplingProb = torch.ones(#self.src)
    self.ppl = nil
  end
end

function SampledDataset:needIndividualLosses()
  return self.sample_w_ppl
end

--[[ initiate sampling ]]
function SampledDataset:sample()
  _G.logger:info('Sampling...')

  -- populate self.samplingProb with self.ppl if average ppl is below self.sample_w_ppl_init

  if self.sample_w_ppl and not self.startedPplSampling then
    local avgPpl = torch.sum(self.ppl)
    avgPpl = avgPpl/self.ppl:size(1)
    if avgPpl < self.sample_w_ppl_init then
      _G.logger:info('Beginning to sample with ppl as probability distribution...')
      self.startedPplSampling = true
    end
  end

  if self.startedPplSampling then

    local threshold = self.sample_w_ppl_max

    if self.sample_w_ppl_max < 0 then
      -- use mode (instead of mean) and stdev of samples with ppl>=mode to
      -- find max ppl to consider (mode + x * stdev). when x is:
      --      x: 1 ~ 100% - 31.7%/2 of train data are not included (divide by 2 because we cut only one-tail)
      --      x: 2 ~ 100% - 4.55%/2
      --      x: 3 ~ 100% - 0.270%/2
      --      x: 4 ~ 100% - 0.00633%/2
      --      x: 5 ~ 100% - 0.0000573%/2
      --      x: 6 ~ 100% - 0.000000197%/2
      --  (https://en.wikipedia.org/wiki/Standard_deviation)
      -- we are using mode instead of average, and only samples above mode to calculate stdev, so
      -- this is not really theoretically valid numbers, but more for emperical uses

      local x = math.abs(self.sample_w_ppl_max)

      -- find mode
      local pplRounded = torch.round(self.ppl)
      local bin = {}
      for i = 1, pplRounded:size(1) do
        if self.ppl[i] ~=  self.sample_w_ppl_init then
          local idx = pplRounded[i]
          if bin[idx] == nil then
            bin[idx] = 0
          end
          bin[idx] = bin[idx] + 1
        end
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
        if self.ppl[i] > mode and self.ppl[i] ~= self.sample_w_ppl_init then
          sum = math.pow(self.ppl[i]-mode, 2)
          cnt = cnt + 1
        end
      end
      local stdev = math.sqrt(sum/(cnt-1))

      threshold = mode + x * stdev

      _G.logger:info('Sampler count: ' .. cnt)
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
  self.sampled = torch.LongTensor(self.samplingSize)
  self.sampled = sampler:batchdraw(self.sampled)
  _G.logger:info('Sampled ' .. self.sampled:size(1) .. ' instances')
end

--[[ get ppl ]]
function SampledDataset:getPpl()
  return self.ppl
end

--[[ set ppl ]]
function SampledDataset:setLoss(batchIdx, loss)
  assert(self:batchCount() >= batchIdx, "Batch idx out of range: " .. batchIdx .. "/" .. self:batchCount())
  local rangeStart = self.maxBatchSize * (batchIdx-1) + 1
  local rangeEnd = math.min(self.maxBatchSize * batchIdx, self:getNumSampled())
  self.ppl[{{rangeStart, rangeEnd}}] = loss:exp()
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

--[[ Return number of sampled instances. ]]
function SampledDataset:getNumSampled()
  return self.sampled:size(1)
end

--[[ Return number of batches. ]]
function SampledDataset:batchCount()
  if self.sampled == nil then
    if #self.src > 0 then
      return 1
    else
      return 0
    end
  end
  return math.ceil(self.sampled:size(1)/self.maxBatchSize)
end

--[[ Get `Batch` number `idx`. If nil make a batch of all the data. ]]
function SampledDataset:getBatch(batchIdx)
  if #self.src == 0 then
    return nil
  end

  if batchIdx == nil or self.sampled == nil then
    return onmt.data.Batch.new(self.src, self.srcFeatures, self.tgt, self.tgtFeatures)
  end

  assert(self:batchCount() >= batchIdx, "Batch idx out of range: " .. batchIdx .. "/" .. self:batchCount())

  local rangeStart = self.maxBatchSize * (batchIdx-1) + 1
  local rangeEnd = math.min(self.maxBatchSize * batchIdx, self:getNumSampled())

  local src = {}
  local tgt

  if self.tgt ~= nil then
    tgt = {}
  end

  local srcFeatures = {}
  local tgtFeatures = {}

  for i = rangeStart, rangeEnd do

    local sampleIdx = self.sampled[i]

    table.insert(src, self.src[sampleIdx])

    if self.srcFeatures[sampleIdx] then
      table.insert(srcFeatures, self.srcFeatures[sampleIdx])
    end

    if self.tgt ~= nil then
      table.insert(tgt, self.tgt[sampleIdx])
      if self.tgtFeatures[sampleIdx] then
        table.insert(tgtFeatures, self.tgtFeatures[sampleIdx])
      end
    end
  end

  return onmt.data.Batch.new(src, srcFeatures, tgt, tgtFeatures)
end

return SampledDataset
