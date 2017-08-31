--[[ Convolutional Encoder similar to the one described in https://arxiv.org/abs/1611.02344.



--]]

local CNNEncoder, parent = torch.class('onmt.CNNEncoder', 'nn.Container')

local options = {
  {
    '-cnn_layers', 2,
    [[Number of convolutional layers in the encoder.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-cnn_kernel', 3,
    [[Kernel size for convolutions. Same in each layer.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-cnn_size', 500,
    [[Number of output units per convolutional layer. Same in each layer.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-use_pos_emb', true,
    [[Add positional embeddings to word embeddings.]],
    {
      structural = 0
    }
  },
  {
    '-max_pos', 50,
    [[Maximum value for positional indexes.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  }

}


function CNNEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


function CNNEncoder:__init(args, inputNetwork)

  parent.__init(self)

  self.inputNet = inputNetwork
  self.args = args

  self.args.numStates = 1

  local convInSize = inputNetwork.inputSize
  local convOutSize = args.cnn_size

  local input = nn.Identity()()
  local sourceSize = nn.Identity()()

  -- Compute input network.
  local inLayer = self.inputNet(input)

  if self.args.use_pos_emb then
    local posEmb = onmt.PositionEmbedding(2, args.max_pos, convInSize)({inLayer, sourceSize})
    inLayer = nn.CAddTable()({inLayer, posEmb})
  end

  local curLayer = inLayer

  for layer_idx=1,args.cnn_layers do

    if self.args.dropout_input or layer_idx > 1 then
      curLayer = nn.Dropout(self.args.dropout)(curLayer)
    end

    local pad = nn.Padding(2,args.cnn_kernel-1)(curLayer) -- right padding
    local conv = nn.TemporalConvolution(convInSize,convOutSize,args.cnn_kernel)(pad)

    -- Add a residual connection, except for the first layer
    if layer_idx > 1 then
      curLayer = nn.CAddTable()({conv, curLayer})
    else
      curLayer = conv
    end

    curLayer = nn.Tanh()(curLayer)

    convInSize = convOutSize
  end

  self:add(nn.gModule({input, sourceSize},{curLayer}))

  self:resetPreallocation()
end


--[[ Return a new CNNEncoder using the serialized data `pretrained`. ]]
function CNNEncoder.load(pretrained)
  local self = torch.factory('onmt.CNNEncoder')()
  parent.__init(self)

  self.modules = pretrained.modules

  self.args = pretrained.args

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function CNNEncoder:serialize()
  return {
    name = 'CNNEncoder',
    modules = self.modules,
    args = self.args
  }
end


function CNNEncoder:resetPreallocation()

  -- Prototype for preallocated state output gradients.
  self.gradStatesOutputProto = torch.Tensor()

end

function CNNEncoder:forward(batch)
  local context = self.modules[1]:forward({batch:getSourceInput(), batch.sourceSize})

  for i=1,batch.size do
    for j=1, batch.sourceLength-batch.sourceSize[i] do
      context[i][j]:fill(0)
    end
  end

  local states = { torch.sum(context, 2):squeeze(2) }

  return states, context
end

function CNNEncoder:backward(batch, gradStatesOutput, gradContextOutput)

  local outputSize = self.args.cnn_size

  if gradStatesOutput then
    self.gradStatesOutputProto = gradStatesOutput[1]
  else
    -- if gradStatesOutput is not defined - start with empty tensor
    self.gradStatesOutputProto = onmt.utils.Tensor.reuseTensor(self.gradStatesOutputProto, { batch.size, outputSize })
  end

  gradContextOutput = gradContextOutput + self.gradStatesOutputProto:view(batch.size,1,outputSize):expandAs(gradContextOutput)

  local gradInputs = self.modules[1]:backward({ batch:getSourceInput(), batch.sourceSize }, gradContextOutput)

  return gradInputs[1]
end
