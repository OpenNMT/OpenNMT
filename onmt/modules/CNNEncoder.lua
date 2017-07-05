--[[ Google's Convolutional Encoder as described in https://arxiv.org/abs/1611.02344.



--]]

local CNNEncoder, parent = torch.class('onmt.CNNEncoder', 'nn.Container')

local options = {
  {
    '-cnn_layers', 5,
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
  }

  -- TODO : attention network
}


function CNNEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


function CNNEncoder:__init(args, inputNetwork)

  parent.__init(self)

  self.inputNet = inputNetwork
  self.args = args

  self.args.numEffectiveLayers = 1

  local convInSize = inputNetwork.inputSize
  local convOutSize = args.cnn_size

  local input = nn.Identity()()

  -- Compute input network.
  local inLayer = self.inputNet(input)

  if self.args.use_pos_emb then
    local posEmb = onmt.PositionEmbedding(2, self.args.preprocess.src_seq_length, self.args.src_word_vec_size[1])(input)
    inLayer = nn.CAddTable()({inLayer, posEmb})
  end

  local curLayer = inLayer

  for layer_idx=1,args.cnn_layers do

    if self.args.dropout_input and layer_idx == 1 then
      curLayer = nn.Dropout(self.args.dropout)(curLayer)
    end

    -- TODO : is there always batch ? which dimension to pad ?
    local pad = nn.Padding(2,1-args.cnn_kernel)(curLayer) -- left padding
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

  local state = nn.Mean(2)(curLayer)

  self:add(nn.gModule({input},{state, curLayer}))

  -- TODO : do we need it ?
  -- self:resetPreallocation()
end


--[[ Return a new CNNEncoder using the serialized data `pretrained`. ]]
function CNNEncoder.load(pretrained)
  local self = torch.factory('onmt.CNNEncoder')()
  parent.__init(self)

  self.modules = pretrained.modules

  self.args = pretrained.args

  -- TODO : do we need it ?
  -- self:resetPreallocation()

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


-- TODO : Do we need it for convolutional decoder ?
-- function Encoder:resetPreallocation()
-- end


function CNNEncoder:forward(batch)
  local output = self.modules[1]:forward(batch:getSourceInput())

  return {output[1]}, output[2]
end

function CNNEncoder:backward(batch, gradStatesOutput, gradContextOutput)

  local gradInputs = self.modules[1]:backward(batch:getSourceInput(), { gradStatesOutput[1], gradContextOutput })
  return gradInputs
end
