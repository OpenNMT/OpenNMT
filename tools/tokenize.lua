require('torch')

local tokenizer = require('tools.utils.tokenizer')
local separators = require('tools.utils.separators')
local BPE = require ('tools.utils.BPE')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**tokenize.lua**")
cmd:text("")

cmd:option('-mode', 'conservative', [[Define how aggressive should the tokenization be - 'aggressive' only keeps sequences of letters/numbers,
  'conservative' allows mix of alphanumeric as in: '2,000', 'E65', 'soft-landing']])
cmd:option('-joiner_annotate', false, [[Include joiner annotation using 'joiner' character]])
cmd:option('-joiner', separators.joiner_marker, [[Character used to annotate joiners]])
cmd:option('-joiner_new', false, [[in joiner_annotate mode, 'joiner' is an independent token]])
cmd:option('-case_feature', false, [[Generate case feature]])
cmd:option('-bpe_model', '', [[Apply Byte Pair Encoding if the BPE model path is given]])

local opt = cmd:parse(arg)

local timer = torch.Timer()
local idx = 1
local bpe

if opt.bpe_model ~= '' then
  bpe = BPE.new(opt.bpe_model, opt.joiner_new)
end

for line in io.lines() do
  local res
  local err

  local tokens

  res, err = pcall(function() tokens = tokenizer.tokenize(opt, line, bpe) end)

  -- it can generate an exception if there are utf-8 issues in the text
  if not res then
    if string.find(err, "interrupted") then
      error("interrupted")
    else
      error("unicode error in line " .. idx .. ": " .. line .. '-' .. err)
    end
  end

  -- output the tokenized and featurized string
  io.write(table.concat(tokens, ' ') .. '\n')

  idx = idx + 1
end

io.stderr:write(string.format('Tokenization completed in %0.3f seconds - %d sentences\n', timer:time().real, idx-1))
