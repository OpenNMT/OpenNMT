--[[ Return the maxLength, sizes, and non-zero count
  of a batch of `seq`s ignoring `ignore` words.
--]]
local function getLength(seq, ignore)
  local sizes = torch.IntTensor(#seq):zero()
  local max = 0

  for i = 1, #seq do
    local len = seq[i]:size(1)
    if ignore ~= nil then
      len = len - ignore
    end
    if max == 0 or len > max then
      max = len
    end
    sizes[i] = len
  end
  return max, sizes
end

--[[ Data management and batch creation.

Batch interface reference [size]:

  * size: number of sentences in the batch [1]
  * sourceLength: max length in source batch [1]
  * sourceSize:  lengths of each source [batch x 1]
  * sourceInput:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * sourceInputFeatures: table of source features sequences
  * targetLength: max length in source batch [1]
  * targetSize: lengths of each source [batch x 1]
  * targetInput: input idx's of target (SABCDEPPPPPP) [batch x max]
  * targetInputFeatures: table of target input features sequences
  * targetOutput: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * targetOutputFeatures: table of target output features sequences

 TODO: change name of size => maxlen
--]]


--[[ A batch of sentences to translate and targets. Manages padding,
  features, and batch alignment (for efficiency).

  Used by the decoder and encoder objects.
--]]
local Batch = torch.class('Batch')

--[[ Create a batch object.

Parameters:

  * `src` - 2D table of source batch indices or prebuilt source batch vectors
  * `srcFeatures` - 2D table of source batch features (opt)
  * `tgt` - 2D table of target batch indices
  * `tgtFeatures` - 2D table of target batch features (opt)
  * `dropoutWords` - words dropout probability
--]]
function Batch:__init(src, srcFeatures, tgt, tgtFeatures)
  src = src or {}
  srcFeatures = srcFeatures or {}
  tgtFeatures = tgtFeatures or {}

  if tgt ~= nil then
    assert(#src == #tgt, "source and target must have the same batch size")
  end

  self.inputVectors = #src > 0 and src[1]:dim() > 1
  self.size = #src
  self.totalSize = self.size -- Used for loss normalization.
  self.sourceLength, self.sourceSize = getLength(src)
  self.sourceInputFeatures = {}

  -- Allocate source tensors.
  if self.inputVectors then
    self.sourceInput = torch.Tensor(self.sourceLength, self.size, src[1]:size(2)):fill(0.0)
  else
    self.sourceInput = torch.LongTensor(self.sourceLength, self.size):fill(onmt.Constants.PAD)

    if #srcFeatures > 0 then
      for _ = 1, #srcFeatures[1] do
        table.insert(self.sourceInputFeatures, self.sourceInput:clone())
      end
    end
  end

  -- Allocate target tensors if defined.
  if tgt ~= nil then
    self.targetLength, self.targetSize = getLength(tgt, 1)

    local targetSeq = torch.LongTensor(self.targetLength, self.size):fill(onmt.Constants.PAD)
    self.targetInput = targetSeq:clone()
    self.targetOutput = targetSeq:clone()

    self.targetInputFeatures = {}
    self.targetOutputFeatures = {}

    if #tgtFeatures > 0 then
      for _ = 1, #tgtFeatures[1] do
        table.insert(self.targetInputFeatures, targetSeq:clone())
        table.insert(self.targetOutputFeatures, targetSeq:clone())
      end
    end
  end

  -- Batch sequences.
  for b = 1, self.size do
    -- Source input is left padded [PPPPPPABCDE].
    local window = {{self.sourceLength - self.sourceSize[b] + 1, self.sourceLength}, b}

    self.sourceInput[window]:copy(src[b])

    for i = 1, #self.sourceInputFeatures do
      self.sourceInputFeatures[i][window]:copy(srcFeatures[b][i])
    end

    if tgt ~= nil then
      local targetLength = tgt[b]:size(1) - 1
      window = {{1, targetLength}, b}

      self.targetInput[window]:copy(tgt[b]:narrow(1, 1, targetLength)) -- [<s>ABCDE]
      self.targetOutput[window]:copy(tgt[b]:narrow(1, 2, targetLength)) -- [ABCDE</s>]

      for i = 1, #self.targetInputFeatures do
        self.targetInputFeatures[i][window]:copy(tgtFeatures[b][i]:narrow(1, 1, targetLength))
        self.targetOutputFeatures[i][window]:copy(tgtFeatures[b][i]:narrow(1, 2, targetLength))
      end
    end
  end
end

--[[ Set source input directly,

Parameters:

  * `sourceInput` - a Tensor of size (sequence_length, batch_size, feature_dim)
  ,or a sequence of size (sequence_length, batch_size). Be aware that sourceInput is not cloned here.

--]]
function Batch:setSourceInput(sourceInput)
  assert (sourceInput:dim() >= 2, 'The sourceInput tensor should be of size (seq_len, batch_size, ...)')
  self.size = sourceInput:size(2)
  self.sourceLength = sourceInput:size(1)
  self.sourceInputFeatures = {}
  self.sourceInput = sourceInput
  self:resetCache()
  return self
end

--[[ Set target input directly.

Parameters:

  * `targetInput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD. Be aware that targetInput is not cloned here.
--]]
function Batch:setTargetInput(targetInput)
  assert (targetInput:dim() == 2, 'The targetInput tensor should be of size (seq_len, batch_size)')
  self.targetInput = targetInput
  self.size = targetInput:size(2)
  self.totalSize = self.size
  self.targetLength = targetInput:size(1)
  self.targetInputFeatures = {}
  self.targetSize = torch.sum(targetInput:transpose(1,2):ne(onmt.Constants.PAD), 2):view(-1):double()
  return self
end

--[[ Set target output directly.

Parameters:

  * `targetOutput` - a tensor of size (sequence_length, batch_size). Padded with onmt.Constants.PAD.  Be aware that targetOutput is not cloned here.
--]]
function Batch:setTargetOutput(targetOutput)
  assert (targetOutput:dim() == 2, 'The targetOutput tensor should be of size (seq_len, batch_size)')
  self.targetOutput = targetOutput
  self.targetOutputFeatures = {}
  return self
end

local function addInputFeatures(inputs, featuresSeq, t)
  local features = {}
  for j = 1, #featuresSeq do
    local feat
    if t > featuresSeq[j]:size(1) then
      feat = onmt.Constants.PAD
    else
      feat = featuresSeq[j][t]
    end
    table.insert(features, feat)
  end
  if #features > 1 then
    table.insert(inputs, features)
  else
    onmt.utils.Table.append(inputs, features)
  end
end

--[[ Get source input batch at timestep `t`. --]]
function Batch:getSourceInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.sourceInput[t]

  if #self.sourceInputFeatures > 0 then
    inputs = { inputs }
    addInputFeatures(inputs, self.sourceInputFeatures, t)
  end

  return inputs
end

--[[ Get target input batch at timestep `t`. --]]
function Batch:getTargetInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.targetInput[t]

  if #self.targetInputFeatures > 0 then
    inputs = { inputs }
    addInputFeatures(inputs, self.targetInputFeatures, t)
  end

  return inputs
end

--[[ Get target output batch at timestep `t` (values t+1). --]]
function Batch:getTargetOutput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local outputs = { self.targetOutput[t] }

  for j = 1, #self.targetOutputFeatures do
    table.insert(outputs, self.targetOutputFeatures[j][t])
  end

  return outputs
end

--[[ Return true if the batch contains sequences of variable lengths. ]]
function Batch:variableLengths()
  if self.variableLen == nil then
    self.variableLen = torch.any(torch.ne(self.sourceSize, self.sourceLength))
  end

  return self.variableLen
end

--[[ Resize the source sequences, adding padding as needed. ]]
function Batch:resizeSource(newLength)
  if newLength == self.sourceLength then
    return
  end

  if newLength < self.sourceLength then
    local lengthDelta = self.sourceLength - newLength
    self.sourceInput = self.sourceInput:narrow(1, lengthDelta + 1, newLength)

    if self.sourceInputFeatures then
      for i = 1, #self.sourceInputFeatures do
        self.sourceInputFeatures[i] = self.sourceInputFeatures[i]
          :narrow(1, lengthDelta + 1, newLength)
      end
    end

    self.sourceSize:cmin(newLength)
  else
    local lengthDelta = newLength - self.sourceLength
    local newSourceInput

    if self.inputVectors then
      newSourceInput =
        self.sourceInput.new(newLength, self.sourceInput:size(2), self.sourceInput:size(3)):fill(0.0)
    else
      newSourceInput =
        self.sourceInput.new(newLength, self.sourceInput:size(2)):fill(onmt.Constants.PAD)

      if self.sourceInputFeatures then
        for i = 1, #self.sourceInputFeatures do
          local newSourceInputFeature = self.sourceInputFeatures[i].new(newLength, self.size)
            :fill(onmt.Constants.PAD)
            :narrow(1, lengthDelta + 1, newLength - lengthDelta)
            :copy(self.sourceInputFeatures[i])
          self.sourceInputFeatures[i] = newSourceInputFeature
        end
      end
    end

    newSourceInput:narrow(1, lengthDelta + 1, newLength - lengthDelta):copy(self.sourceInput)
    self.sourceInput = newSourceInput
  end

  self.sourceLength = newLength
  self:resetCache()
end

function Batch:resetCache()
  self.sourceInputRev = nil
  self.sourceInputFeaturesRev = nil
  self.variableLen = nil
end

function Batch:reverseSourceInPlace()
  if not self.sourceInputRev then
    self.sourceInputRev = self.sourceInput:clone()
    if self.sourceInputFeatures then
      self.sourceInputFeaturesRev = onmt.utils.Tensor.recursiveClone(self.sourceInputFeatures)
    end

    for b = 1, self.size do
      local reversedIndices = torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long()
      local window = {{self.sourceLength - self.sourceSize[b] + 1, self.sourceLength}, b}

      self.sourceInputRev[window]:copy(self.sourceInput[window]:index(1, reversedIndices))

      if self.sourceInputFeatures then
        for i = 1, #self.sourceInputFeatures do
          self.sourceInputFeaturesRev[i][window]
            :copy(self.sourceInputFeatures[i][window]:index(1, reversedIndices))
        end
      end
    end
  end

  self.sourceInput, self.sourceInputRev = self.sourceInputRev, self.sourceInput
  self.sourceInputFeatures, self.sourceInputFeaturesRev = self.sourceInputFeaturesRev, self.sourceInputFeatures
end

function Batch:dropoutWords(p)
  assert(not self.inputVectors, "-dropout_words option cannot be used with input vectors")

  local vocabMask = torch.Tensor()

  for i = 1, batch.sourceInput:size(1) do
    local vocab = {}
    local vocabMap = {}

    for j = 1, batch.sourceInput:size(2) do
      local x = batch.sourceInput[i][j]
      if x > onmt.Constants.EOS and not vocab[x] then
        table.insert(vocabMap, x)
        vocab[x] = #vocabMap
      end
    end

    vocabMask:resize(#vocabMap)
    vocabMask:bernoulli(1-p)

    for j = 1, batch.sourceInput:size(2) do
      local x = batch.sourceInput[i][j]
      if x > onmt.Constants.EOS and vocabMask[vocab[x]] == 0 then
        batch.sourceInput[i][j] = onmt.Constants.PAD
      end
    end
  end
end

return Batch
