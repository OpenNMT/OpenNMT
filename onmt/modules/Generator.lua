--[[ Default decoder generator. Given RNN state, produce categorical distribution for tokens and features

     Simply implements $$softmax(W h b)$$.
--]]
local Generator, parent = torch.class('onmt.Generator', 'onmt.Network')

-- for back compatibility - still declare FeaturesGenerator - but no need to define it
torch.class('onmt.FeaturesGenerator', 'onmt.Network')

function Generator:__init(opt, dicts, sizes)
  parent.__init(self)
  self:buildGenerator(opt, dicts, sizes)
end

--[[ Build or convert a generator - if convert is true, keep the current generator linear parameters.]]
function Generator:buildGenerator(opt, dicts, sizes, convert)
  local rnn_size = opt.rnn_size
  if convert then
    local prevgen = (self.needOutput and 'NCE') or 'NLL'
    local newgen = (opt.criterion == 'nce' and 'NCE') or 'NLL'
    if newgen == prevgen then
      return
    end
    _G.logger:info(' * Converting '..prevgen..' criterion into '..newgen)
    if not sizes then
      sizes = {}
      for i = 1, #self.net.modules do
        local m = self.net.modules[i]
        -- m is a Sequential - m.modules[1] is a selector, m.modules[2] is NCEModule or Linear
        table.insert(sizes, m.modules[2].weight:size(1))
        rnn_size = m.modules[2].weight:size(2)
      end
    end
  end

  local generator = nn.ConcatTable()

  local selectInput = nn.Identity()

  if opt.criterion == 'nce' then
    self.needOutput = true
    selectInput = nn.SelectTable(1)
  else
    self.needOutput = false
  end

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

--[[ Release Generator - ie. if NCE convert to Linear/Logsoftmax. ]]
function Generator:release()
  local opt = { criterion='nll' }
  self:buildGenerator(opt, nil, nil, true)
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
