--[[ PDBiEncoder is a pyramidal deep bidirectional sequencer used for the source language.

The outputs of each bidirectional layer is merged to reduce the time dimension.

]]
local PDBiEncoder, parent = torch.class('onmt.PDBiEncoder', 'nn.Container')

local options = {
  {
    '-pdbrnn_reduction', 2,
    [[Time-reduction factor at each layer.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isInt(1),
      structural = 0
    }
  },
  {
    '-pdbrnn_merge', 'concat',
    [[Merge action when reducing time.]],
    {
      enum = {'concat', 'sum'},
      structural = 0
    }
  }
}

function PDBiEncoder.declareOpts(cmd)
  onmt.DBiEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


--[[ Create a pyramidal deep bidirectional encoder.

Parameters:

  * `args` - global arguments
  * `input` - input neural network
]]
function PDBiEncoder:__init(args, input)
  parent.__init(self)

  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  self.args.multiplier = math.pow(self.args.pdbrnn_reduction, args.layers - 1)
  self.args.hiddenSize = args.rnn_size
  self.args.numEffectiveLayers = 0

  for i = 1, args.layers do
    local layerArgs = onmt.utils.Tensor.deepClone(args)
    layerArgs.layers = 1

    if i > 1 then
      input = nn.Identity()

      if self.args.pdbrnn_merge == 'sum' then
        input.inputSize = args.rnn_size
      elseif self.args.pdbrnn_merge == 'concat' then
        input.inputSize = args.rnn_size * self.args.pdbrnn_reduction
      end

      -- Rely on input dropout for subsequent layers.
      if args.dropout > 0 then
        layerArgs.dropout_input = true
      end
    end

    local brnn = onmt.BiEncoder(layerArgs, input)
    self.args.numEffectiveLayers = self.args.numEffectiveLayers + brnn.args.numEffectiveLayers
    self:add(brnn)
  end

  self:resetPreallocation()
end

--[[ Return a new PDBiEncoder using the serialized data `pretrained`. ]]
function PDBiEncoder.load(pretrained, className)
  local self = torch.factory(className or 'onmt.PDBiEncoder')()
  parent.__init(self)

  for i = 1, #pretrained.modules do
    self:add(onmt.BiEncoder.load(pretrained.modules[i]))
  end

  self.args = pretrained.args

  self:resetPreallocation()
  return self
end

--[[ Return data to serialize. ]]
function PDBiEncoder:serialize(className)
  local modulesData = {}
  for i = 1, #self.modules do
    table.insert(modulesData, self.modules[i]:serialize())
  end

  return {
    name = className or 'PDBiEncoder',
    modules = modulesData,
    args = self.args
  }
end

function PDBiEncoder:resetPreallocation()
  -- Prototype for preallocated full context vector.
  self.contextProto = torch.Tensor()

  -- Prototype for preallocated full hidden states tensors.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated gradient context
  self.gradContextProto = torch.Tensor()
end

function PDBiEncoder:maskPadding()
  for i, layer in ipairs(self.modules) do
    if i == 1 or self.args.pdbrnn_reduction == 1 then
      layer:maskPadding()
    end
  end
end

-- size of context vector
function PDBiEncoder:contextSize(sourceSize, sourceLength)
  local contextLength = math.ceil(sourceLength / self.args.multiplier)
  local contextSize

  if type(sourceSize) == 'table' then
    contextSize = {}
    for i = 1, #sourceSize do
      table.insert(contextSize, math.ceil(sourceSize[i] / self.args.multiplier))
    end
  elseif type(sourceSize) == 'int' then
    contextSize = math.ceil(sourceSize / self.args.multiplier)
  else
    contextSize = torch.ceil(sourceSize / self.args.multiplier)
  end

  return contextSize, contextLength
end

function PDBiEncoder:forward(batch)
  -- Make source length divisible by the total reduction.
  batch.sourceLength = math.ceil(batch.sourceLength / self.args.multiplier) * self.args.multiplier

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.hiddenSize })
  end

  local finalStates = onmt.utils.Tensor.reuseTensorTable(self.statesProto,
                                                         { batch.size, self.args.hiddenSize })
  local finalContext
  local statesIdx = 1

  self.inputs = { batch }
  self.statesRange = {}

  for i = 1, #self.modules do
    local states, context = self.modules[i]:forward(self.inputs[i])

    -- Save layer states.
    table.insert(self.statesRange, { statesIdx, #states })
    for j = 1, #states do
      finalStates[statesIdx]:copy(states[j])
      statesIdx = statesIdx + 1
    end

    if i == #self.modules then
      finalContext = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                   context:size())
      finalContext:copy(context)
    else
      local nextContext

      -- Reduce time dimension.
      if self.args.pdbrnn_reduction == 1 then
        nextContext = context
      elseif self.args.pdbrnn_merge == 'sum' then
        nextContext = context
          :view(context:size(1),
                self.args.pdbrnn_reduction,
                context:size(2) / self.args.pdbrnn_reduction,
                context:size(3))
          :sum(2)
          :squeeze(2)
      elseif self.args.pdbrnn_merge == 'concat' then
        nextContext = context
          :view(context:size(1),
                context:size(2) / self.args.pdbrnn_reduction,
                context:size(3) * self.args.pdbrnn_reduction)
      end

      table.insert(self.inputs, onmt.data.BatchTensor.new(nextContext, batch.sourceSize))
    end
  end

  return finalStates, finalContext
end

function PDBiEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  local gradInputs

  for i = #self.modules, 1, -1 do
    local layerGradStatesOutput
    if gradStatesOutput then
      layerGradStatesOutput = onmt.utils.Table.subrange(gradStatesOutput,
                                                        self.statesRange[i][1],
                                                        self.statesRange[i][2])
    end

    gradInputs = self.modules[i]:backward(self.inputs[i],
                                          layerGradStatesOutput,
                                          gradContextOutput)

    if i ~= 1 then
      gradContextOutput = onmt.utils.Tensor.reuseTensor(
        self.gradContextProto,
        { batch.size, #gradInputs * self.args.pdbrnn_reduction, self.args.hiddenSize })

      for t = 1, #gradInputs do
        if self.args.pdbrnn_reduction == 1 then
          gradContextOutput[{{}, t}]:copy(gradInputs[t])
        elseif self.args.pdbrnn_merge == 'sum' then
          -- After a sum operation, just replicate the gradients.
          for j = 1, self.args.pdbrnn_reduction do
            gradContextOutput[{{}, (t - 1) * self.args.pdbrnn_reduction + j}]:copy(gradInputs[t])
          end
        elseif self.args.pdbrnn_merge == 'concat' then
          gradContextOutput
            :narrow(2, (t - 1) * self.args.pdbrnn_reduction + 1, self.args.pdbrnn_reduction)
            :copy(gradInputs[t])
        end
      end
    end
  end

  return gradInputs
end
