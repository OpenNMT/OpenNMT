--[[ Default decoder generator. Given RNN state, produce categorical distribution.

Simply implements $$softmax(W h + b)$$.
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')


function Generator:__init(opt, dicts, sizes)
  parent.__init(self, self:_buildGenerator(opt, dicts, sizes))
end

function Generator:_buildGenerator(opt, dicts, sizes)
  local generator = nn.ConcatTable()

  local selectInput = nn.Identity()

  if opt.criterion == 'nce' then
    self.needOutput = 1
    selectInput = nn.SelectTable(1)
  end

  for i = 1, #sizes do
    local feat_generator
    if i == 1 and opt.criterion == 'nce' then
      assert(dicts.words.freqTensor, "missing frequencies in dictionary - use -keep_frequency in preprocess.lua")
      assert(onmt.NCEModule, "missing NCE module - install dpnn torch libraries")
      local selectInputOutput = nn.ConcatTable()
                          :add(nn.SelectTable(1)) -- first element is the input
                          :add(nn.Sequential():add(nn.SelectTable(2)):add(nn.SelectTable(i)))

      feat_generator = nn.Sequential()
                    :add(selectInputOutput)
                    :add(onmt.NCEModule(opt, opt.rnn_size, sizes[i], dicts.words.freqTensor))
    else
      feat_generator = nn.Sequential()
                    :add(selectInput)
                    :add(nn.Linear(opt.rnn_size, sizes[i]))
                    :add(nn.LogSoftMax())
    end
    generator:add(feat_generator)
  end
  return generator
end

--[[ Release Generator - if NCE convert to Linear/Logsoftmax. ]]
function Generator:release()
  if self.needOutput then
    _G.logger:info(' * Converting NCE module into regular linear/softmax')
    local generator = nn.ConcatTable()
    for i = 1, #self.net.modules do
      local m = self.net.modules[i]
      -- m is a Sequential - m.modules[1] is a selector, m.modules[2] is NCEModule or Linear
      local linear = nn.Linear(m.modules[2].weight:size(2),m.modules[2].weight:size(1))
      linear.weight = m.modules[2].weight
      linear.bias = m.modules[2].bias
      local feat_generator = nn.Sequential()
                          :add(linear)
                          :add(nn.LogSoftMax())
      generator:add(feat_generator)
    end
    self.net = generator
  end
end

function Generator:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
