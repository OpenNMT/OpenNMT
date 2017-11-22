local unicode = require('tools.utils.unicode')

local myopt =
{
  {
    '-mode', 'conservative',
    [[Define how aggressive should the tokenization be. `aggressive` only keeps sequences
      of letters/numbers, `conservative` allows a mix of alphanumeric as in: "2,000", "E65",
      "soft-landing", etc. `space` is doing space tokenization. `char` is doing character tokenization]],
    {
      enum = {'space', 'conservative', 'aggressive', 'char'}
    }
  }
}

local function declareOptsFn(cmd)
  cmd:setCmdLineOptions(myopt, 'Tokenizer')
end

local function mytokenization(opt, line)
  -- fancy tokenization, it has to return a table of tokens (possibly with features)
  if opt.mode == "char" then
    local tokens = {}
    for v, c, _ in unicode.utf8_iter(line) do
      if unicode.isSeparator(v) then
        table.insert(tokens, '_')
      else
        table.insert(tokens, c)
      end
    end
    return tokens
  end
end

return {
  tokenize = mytokenization,
  declareOpts = declareOptsFn
}
