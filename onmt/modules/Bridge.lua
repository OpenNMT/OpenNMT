--[[ Bridge between encoder and decoder. ]]
local Bridge, parent = torch.class('onmt.Bridge', 'onmt.Network')

local options = {
  {
    '-bridge', 'copy',
    [[Define how to pass encoder states to the decoder. With `copy`, the encoder and decoder
      must have the same number of layers.]],
    {
      enum = {'copy', 'dense', 'dense_nonlinear', 'last', 'none'},
      structural = 0
    }
  }
}

function Bridge.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

function Bridge:__init(bridgeType, encoderRnnSize, encoderNumStates, decoderRnnSize, decoderNumStates)
  _G.logger:info(' * Bridge: %s', bridgeType)

  local bridge

  if bridgeType == 'copy' then
    assert(encoderRnnSize == decoderRnnSize and encoderNumStates == decoderNumStates,
           'with the copy bridge type, encoder and decoder must have the same number of layers and hidden size')

    bridge = nn.MapTable(nn.Identity())
  elseif bridgeType == 'dense' or bridgeType == 'dense_nonlinear' then
    bridge = nn.Sequential()
      :add(nn.JoinTable(2))

    bridge:add(nn.Linear(encoderRnnSize * encoderNumStates, decoderRnnSize * decoderNumStates, false))
    if bridgeType == 'dense_nonlinear' then
      bridge:add(nn.Tanh())
    end

    bridge:add(nn.View(-1, decoderNumStates, decoderRnnSize))
    bridge:add(nn.SplitTable(2))
  elseif bridgeType == 'last' then
    assert(encoderRnnSize == decoderRnnSize and encoderNumStates > decoderNumStates,
           'with the `last` bridge type, encoder and decoder must have same rnn size, and encoder should'
           .. ' have more layers')
    bridge = nn.NarrowTable(encoderNumStates-decoderNumStates+1, decoderNumStates)
  elseif bridgeType == 'none' then
    bridge = nil
  else
    error('invalid bridge type: ' .. bridgeType)
  end

  parent.__init(self, bridge)
end

function Bridge.load(pretrained)
  if pretrained then
    return pretrained
  else
    -- For backward compatibility create the default bridge.
    local self = torch.factory('onmt.Bridge')()
    parent.__init(self, nn.MapTable(nn.Identity()))
    return self
  end
end

function Bridge:updateOutput(input)
  if self.net then
    return parent.updateOutput(self, input)
  end
end

function Bridge:updateGradInput(input, gradOutput)
  if self.net then
    return parent.updateGradInput(self, input, gradOutput)
  else
    self.gradInput = nil
    return self.gradInput
  end
end

function Bridge:accGradParameters(input, gradOutput, scale)
  if self.net then
    self.net:accGradParameters(input, gradOutput, scale)
  end
end
