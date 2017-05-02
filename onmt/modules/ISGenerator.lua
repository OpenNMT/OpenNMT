--[[
    Importance Sampling generator
--]]
local ISGenerator, parent = torch.class('onmt.ISGenerator', 'onmt.Generator')

local options = {
  {
    '-importance_sampling_tgt_voc_size', 0,
    [[Use importance sampling approach as approximation of full softmax.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      depends = function(opt) return opt.importance_sampling_tgt_voc_size == 0 or
                                     opt.sample ~= 0, "requires '-sample' option" end
    }
  }
}

function ISGenerator.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Generator')
end

function ISGenerator:__init(opt, sizes)
  parent.__init(self, opt, sizes)
end

function ISGenerator:_buildGenerator(opt, sizes)
  local generator = nn.ConcatTable()
  local rnn_size = opt.rnn_size

  for i = 1, #sizes do
    local feat_generator
    local linear
    if i == 1 then
      linear = onmt.RIndexLinear(rnn_size, sizes[i])
      self.rindexLinear = linear
    else
      linear = nn.Linear(rnn_size, sizes[i])
    end
    feat_generator = nn.Sequential()
                        :add(nn.SelectTable(1))
                        :add(linear)
                        :add(nn.LogSoftMax())
    generator:add(feat_generator)
  end

  self:set(generator)
end

--[[ If the target vocabulary for the batch is not full vocabulary ]]
function ISGenerator:setTargetVoc(t)
  self.rindexLinear:setOutputIndices(t)
end

--[[ Release Generator for inference only ]]
function ISGenerator:release()
end

function ISGenerator:updateOutput(input)
  input = type(input) == 'table' and input or { input }
  self.output = self.net:updateOutput(input)
  return self.output
end

function ISGenerator:updateGradInput(input, gradOutput)
  input = type(input) == 'table' and input or { input }
  self.gradInput = self.net:updateGradInput(input, gradOutput)[1]
  return self.gradInput
end

function ISGenerator:accGradParameters(input, gradOutput, scale)
  input = type(input) == 'table' and input or { input }
  self.net:accGradParameters(input, gradOutput, scale)
end
