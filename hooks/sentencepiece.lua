local sentencepiece = require('lua-sentencepiece')

local tokenizer = require('tools.utils.tokenizer')

local myopt =
{
  {
    '-mode', 'conservative',
    [[Define how aggressive should the tokenization be. `aggressive` only keeps sequences
      of letters/numbers, `conservative` allows a mix of alphanumeric as in: "2,000", "E65",
      "soft-landing", etc. `space` is doing space tokenization. `none` is not doing tokenization,
      has to be used with sentencepiece.]],
    {
      enum = {'space', 'conservative', 'aggressive', 'none'}
    }
  },
  {
    '-sentencepiece', '',
    [[Path to the model to use with sentencepiece - can be combined with regular tokenization mode.]]
  }
}

local function declareOptsFn(cmd)
  cmd:setCmdLineOptions(myopt, 'Tokenizer')
end

local function mytokenization(opt, line, bpe)
  if opt.sentencepiece ~= '' then
    assert(not opt.case_feature, "sentence piece is not compatible with case_feature")
    local pretok = line

    if opt.mode ~= 'none' then
      local save_hooks = _G.hookManager.hooks
      _G.hookManager.hooks = {}
      pretok = table.concat(tokenizer.tokenize(opt, line, bpe), " ")
      _G.hookManager.hooks = save_hooks
    end

    local tokens = sentencepiece.encode(opt.sentencepiece, pretok)

    return tokens
  end
end

local function split(str, sep)
  local res = {}
  local index = 1

  while index <= str:len() do
    local sepStart, sepEnd = str:find(sep, index)

    local sub
    if not sepStart then
      sub = str:sub(index)
      table.insert(res, sub)
      index = str:len() + 1
    else
      sub = str:sub(index, sepStart - 1)
      table.insert(res, sub)
      index = sepEnd + 1
      if index > str:len() then
        table.insert(res, '')
      end
    end
  end

  return res
end

local function mydetokenization(line, opt)
  if opt.sentencepiece ~= '' then
    local tokens = split(line, ' ')
    local str = sentencepiece.decode(opt.sentencepiece, tokens)
    if str:find("■") then
      local save_hooks = _G.hookManager.hooks
      _G.hookManager.hooks = {}
      str = tokenizer.detokenize(str, {joiner="■"})
      _G.hookManager.hooks = save_hooks
    end
    return str
  end
end

return {
  tokenize = mytokenization,
  detokenize = mydetokenization,
  hookName = function() return "sentencepiece" end,
  declareOpts = declareOptsFn
}
