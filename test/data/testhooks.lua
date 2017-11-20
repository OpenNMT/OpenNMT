local options_sampledvocab = {
  {
    '-sample_vocab_type', 'beet',
    [[Type of sample vocabulary]],
    {
      enum = {'beet', 'carrot'}
    }
  },
  {
    '-sample_vocab', true,
    [[now activated by default]]
  }
}

-- remove an option, just set empty documentation
local other_options = {
  {
  '-profiler', '__REMOVE__', [[]]
  }
}

local jean_options = {
  {
    '-happy', false,
    [[Are you happy?]]
  }
}

local function declareOptsFn(cmd)
  cmd:setCmdLineOptions(options_sampledvocab, 'Sampled Vocabulary')
  cmd:setCmdLineOptions(jean_options, 'Jean\'s')
  cmd:setCmdLineOptions(other_options, 'Other')
end

local function tokenizeFn()
  return "XX"
end

return {
  declareOpts = declareOptsFn,
  tokenize = tokenizeFn
}
