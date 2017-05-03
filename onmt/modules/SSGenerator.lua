--[[ Default decoder generator. Given RNN state, produce categorical distribution for tokens and features

     Simply implements $$softmax(W h b)$$.

     version 2: merge FeaturesGenerator and Generator - the generator nn is a table
--]]
local Generator, parent = torch.class('onmt.SSGenerator', 'onmt.Generator')

function SSGenerator:__init(opt, dicts, sizes)
  parent.__init(self)
end

--[[ Build or convert a generator - if convert is true, keep the current generator linear parameters.]]
function SSGenerator:_buildGenerator(opt, dicts, sizes)
  local rnn_size = opt.rnn_size

  local generator = nn.ConcatTable()

  local selectInput = nn.SelectTable(1)

  for i = 1, #sizes do
    local feat_generator
    local linmod
    if i == 1 and opt.criterion == 'nce' then
      assert(dicts.words.freqTensor, "missing frequencies in dictionary - use -keep_frequency in preprocess.lua")
      assert(onmt.NCEModule, "missing NCE module - install dpnn torch libraries")
      local selectInputOutput = nn.ConcatTable()
                          :add(nn.SelectTable(1)) -- first element is the input
                          :add(nn.Sequential():add(nn.SelectTable(2)):add(nn.SelectTable(i)))

      linmod = onmt.NCEModule(opt, rnn_size, sizes[i], dicts.words.freqTensor)

      feat_generator = nn.Sequential()
                    :add(selectInputOutput)
                    :add(linmod)

    else
      linmod = nn.Linear(rnn_size, sizes[i])
      feat_generator = nn.Sequential()
                    :add(selectInput)
                    :add(linmod)
                    :add(nn.LogSoftMax())
    end
    if convert then
      linmod.weight = self.net.modules[i].modules[2].weight
      linmod.bias = self.net.modules[i].modules[2].bias
    end
    generator:add(feat_generator)
  end
  self:set(generator)
end

function Generator:updateOutput(input)
  input = type(input) == 'table' and input or { input }
  self.output = self.net:updateOutput(input)
  return self.output
end

function Generator:updateGradInput(input, gradOutput)
  input = type(input) == 'table' and input or { input }
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Generator:accGradParameters(input, gradOutput, scale)
  input = type(input) == 'table' and input or { input }
  self.net:accGradParameters(input, gradOutput, scale)
end
