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
