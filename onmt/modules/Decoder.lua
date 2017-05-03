--[[ Unit to decode a sequence of output tokens.

     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local Decoder, parent = torch.class('onmt.Decoder', 'onmt.Sequencer')

local options = {
  {
    '-input_feed', true,
    [[Feed the context vector at each time step as additional input
      (via concatenation with the word embeddings) to the decoder.]],
    {
      structural = 0
    }
  }
}

function Decoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
  onmt.Encoder.declareOpts(cmd)
end

--[[ Construct a decoder layer.

Parameters:

  * `args` - module arguments
  * `inputNetwork` - input nn module.
  * `generator` - an output generator.
  * `attentionModel` - attention model to apply on source.
--]]
function Decoder:__init(args, inputNetwork, generator, attentionModel)
  local RNN = onmt.LSTM
  if args.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  local inputSize = inputNetwork.inputSize
  if args.input_feed then
    inputSize = inputSize + args.rnn_size
  end

  local rnn = RNN.new(args.layers, inputSize, args.rnn_size,
                      args.dropout, args.residual, args.dropout_input)

  self.rnn = rnn
  self.inputNet = inputNetwork

  self.args = args
  self.args.rnnSize = self.rnn.outputSize
  self.args.numEffectiveLayers = self.rnn.numEffectiveLayers

  self.args.inputIndex = {}
  self.args.outputIndex = {}

  self.args.inputFeed = args.input_feed

  parent.__init(self, self:_buildModel(attentionModel))

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generator = generator
  self:add(self.generator)

  self:resetPreallocation()
end

function Decoder:returnIndividualLosses(enable)
  self.indvLoss = enable
end

--[[ Return a new Decoder using the serialized data `pretrained`. ]]
function Decoder.load(pretrained)
  local self = torch.factory('onmt.Decoder')()

  self.args = pretrained.args

  parent.__init(self, pretrained.modules[1])
  self.generator = onmt.Generator.load(pretrained.modules[2])
  self:add(self.generator)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function Decoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

function Decoder:resetPreallocation()
  if self.args.inputFeed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end

--[[ Build a default one time-step of the decoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.
--]]
function Decoder:_buildModel(attentionModel)
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, self.args.numEffectiveLayers do
    local h0 = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)
  self.args.inputIndex.x = #inputs

  local context = nn.Identity()() -- batchSize x sourceLength x rnnSize
  table.insert(inputs, context)
  self.args.inputIndex.context = #inputs

  local inputFeed
  if self.args.inputFeed then
    inputFeed = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, inputFeed)
    self.args.inputIndex.inputFeed = #inputs
  end

  -- Compute the input network.
  local input = self.inputNet(x)

  -- If set, concatenate previous decoder output.
  if self.args.inputFeed then
    input = nn.JoinTable(2)({input, inputFeed})
  end
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.rnn(states)

  if self.args.numEffectiveLayers > 1 then
    -- The output of a subgraph is a node: split it to access the last RNN output.
    outputs = { outputs:split(self.args.numEffectiveLayers) }
  else
    outputs = { outputs }
  end

  -- Compute the attention here using h^L as query.
  local attnLayer = attentionModel
  attnLayer.name = 'decoderAttn'
  local attnInput = {outputs[#outputs], context}
  local attnOutput = attnLayer(attnInput)
  if self.rnn.dropout > 0 then
    attnOutput = nn.Dropout(self.rnn.dropout)(attnOutput)
  end
  table.insert(outputs, attnOutput)
  return nn.gModule(inputs, outputs)
end

function Decoder:findAttentionModel()
  if not self.decoderAttn then
    self.network:apply(function (layer)
      if layer.name == 'decoderAttn' then
        self.decoderAttn = layer
      elseif layer.name == 'softmaxAttn' then
        self.softmaxAttn = layer
      end
    end)
    self.decoderAttnClones = {}
  end
  for t = #self.decoderAttnClones+1, #self.networkClones do
    self:net(t):apply(function (layer)
      if layer.name == 'decoderAttn' then
        self.decoderAttnClones[t] = layer
      elseif layer.name == 'softmaxAttn' then
        self.decoderAttnClones[t].softmaxAttn = layer
      end
    end)
  end
end

--[[ Mask padding means that the attention-layer is constrained to
  give zero-weight to padding. This is done by storing a reference
  to the softmax attention-layer.

  Parameters:

  * See  [onmt.MaskedSoftmax](onmt+modules+MaskedSoftmax).
--]]
function Decoder:maskPadding(sourceSizes, sourceLength)
  self:findAttentionModel()

  local function substituteSoftmax(module)
    if module.name == 'softmaxAttn' then
      local mod
      if sourceSizes ~= nil then
        mod = onmt.MaskedSoftmax(sourceSizes, sourceLength)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmaxAttn'
      mod:type(module._type)
      self.softmaxAttn = mod
      return mod
    else
      return module
    end
  end

  self.decoderAttn:replace(substituteSoftmax)

  for t = 1, #self.networkClones do
    self.decoderAttnClones[t]:replace(substituteSoftmax)
  end
end

--[[ Run one step of the decoder.

Parameters:

  * `input` - input to be passed to inputNetwork.
  * `prevStates` - stack of hidden states (batch x layers*model x rnnSize)
  * `context` - encoder output (batch x n x rnnSize)
  * `prevOut` - previous distribution (batch x #words)
  * `t` - current timestep

Returns:

 1. `out` - Top-layer hidden state.
 2. `states` - All states.
--]]
function Decoder:forwardOne(input, prevStates, context, prevOut, t)
  local inputs = {}

  -- Create RNN input (see sequencer.lua `buildNetwork('dec')`).
  onmt.utils.Table.append(inputs, prevStates)
  table.insert(inputs, input)
  table.insert(inputs, context)
  local inputSize
  if torch.type(input) == 'table' then
    inputSize = input[1]:size(1)
  else
    inputSize = input:size(1)
  end

  if self.args.inputFeed then
    if prevOut == nil then
      table.insert(inputs, onmt.utils.Tensor.reuseTensor(self.inputFeedProto,
                                                         { inputSize, self.args.rnnSize }))
    else
      table.insert(inputs, prevOut)
    end
  end

  -- Remember inputs for the backward pass.
  if self.train then
    self.inputs[t] = inputs
  end

  local outputs = self:net(t):forward(inputs)

  -- Make sure decoder always returns table.
  if type(outputs) ~= "table" then outputs = { outputs } end

  local out = outputs[#outputs]
  local states = {}
  for i = 1, #outputs - 1 do
    table.insert(states, outputs[i])
  end

  return out, states
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - `Batch` object
  * `initialStates` - initialization of decoder.
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function Decoder:forwardAndApply(batch, initialStates, context, func)
  -- TODO: Make this a private method.

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(#initialStates,
                                                         self.stateProto,
                                                         { batch.size, self.args.rnnSize })
  end

  local states = onmt.utils.Tensor.copyTensorTable(self.statesProto, initialStates)

  local prevOut

  for t = 1, batch.targetLength do
    prevOut, states = self:forwardOne(batch:getTargetInput(t), states, context, prevOut, t)
    func(prevOut, t)
  end
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - a `Batch` object.
  * `initialStates` - a batch of initial decoder states (optional) [0]
  * `context` - the context to apply attention to.

  Returns: Table of top hidden state for each timestep.
--]]
function Decoder:forward(batch, initialStates, context)
  initialStates = initialStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })
  if self.train then
    self.inputs = {}
  end

  local outputs = {}

  self:forwardAndApply(batch, initialStates, context, function (out)
    table.insert(outputs, out)
  end)

  return outputs
end

--[[ Compute the backward update.

Parameters:

  * `batch` - a `Batch` object
  * `outputs` - expected outputs
  * `criterion` - a single target criterion object

  Note: This code runs both the standard backward and criterion forward/backward.
  It returns both the gradInputs and the loss.
  -- ]]
function Decoder:backward(batch, outputs, criterion)
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers + 1,
                                                              self.gradOutputProto,
                                                              { batch.size, self.args.rnnSize })
  end

  local gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                             { batch.size, self.args.rnnSize })
  local gradContextInput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                                         { batch.size, batch.encoderOutputLength or batch.sourceLength, self.args.rnnSize })

  local loss = 0
  local indvAvgLoss = torch.zeros(outputs[1]:size(1))

  for t = batch.targetLength, 1, -1 do
    local refOutput = batch:getTargetOutput(t)

    -- sampling-based generator need outputs during training
    local prepOutputs = { outputs[t], refOutput }

    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    local pred = self.generator:forward(prepOutputs)

    if self.indvLoss then
      for i = 1, pred[1]:size(1) do
        if t <= batch.targetSize[i] then
          local tmpPred = {}
          local tmpOutput = {}
          for j = 1, #pred do
            table.insert(tmpPred, pred[j][{{i}, {}}])
            table.insert(tmpOutput, refOutput[j][{{i}}])
          end
          local tmpLoss = criterion:forward(tmpPred, tmpOutput)
          indvAvgLoss[i] = indvAvgLoss[i] + tmpLoss
          loss = loss + tmpLoss
        end
      end
    else
      loss = loss + criterion:forward(pred, refOutput)
    end

    -- Compute the criterion gradient.
    local genGradOut = criterion:backward(pred, refOutput)

    -- normalize gradient - we might have several batches in parallel, so we divide by total size of batch
    for j = 1, #genGradOut do
      -- each criterion might have its own normalization function
      if criterion.normalizationFunc and criterion.normalizationFunc[j] ~= false then
        criterion.normalizationFunc[j](genGradOut[j], batch.totalSize)
      else
        genGradOut[j]:div(batch.totalSize)
      end
    end

    -- Compute the final layer gradient.
    local decGradOut = self.generator:backward(prepOutputs, genGradOut)

    gradStatesInput[#gradStatesInput]:add(decGradOut)

    -- Compute the standard backward.
    local gradInput = self:net(t):backward(self.inputs[t], gradStatesInput)

    -- Accumulate encoder output gradients.
    gradContextInput:add(gradInput[self.args.inputIndex.context])
    gradStatesInput[#gradStatesInput]:zero()

    -- Accumulate previous output gradients with input feeding gradients.
    if self.args.inputFeed and t > 1 then
      gradStatesInput[#gradStatesInput]:add(gradInput[self.args.inputIndex.inputFeed])
    end

    -- Prepare next decoder output gradients.
    for i = 1, #self.statesProto do
      gradStatesInput[i]:copy(gradInput[i])
    end
  end

  if self.indvLoss then
    indvAvgLoss = torch.cdiv(indvAvgLoss, batch.targetSize:double())
  end

  return gradStatesInput, gradContextInput, loss, indvAvgLoss
end

--[[ Compute the loss on a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `initialStates` - initialization of decoder.
  * `context` - the attention context.
  * `criterion` - a pointwise criterion.

--]]
function Decoder:computeLoss(batch, initialStates, context, criterion)
  initialStates = initialStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local loss = 0
  self:forwardAndApply(batch, initialStates, context, function (out, t)
    local pred = self.generator:forward(out)
    local refOutput = batch:getTargetOutput(t)
    loss = loss + criterion:forward(pred, refOutput)
  end)

  return loss
end


--[[ Compute the score of a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `initialStates` - initialization of decoder.
  * `context` - the attention context.

--]]
function Decoder:computeScore(batch, initialStates, context)
  initialStates = initialStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local score = {}

  self:forwardAndApply(batch, initialStates, context, function (out, t)
    local pred = self.generator:forward(out)
    for b = 1, batch.size do
      if t <= batch.targetSize[b] then
        score[b] = (score[b] or 0) + pred[1][b][batch.targetOutput[t][b]]
      end
    end
  end)

  return score
end
