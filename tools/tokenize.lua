require('torch')
require('onmt.init')

local threads = require 'threads'
local tokenizer = require('tools.utils.tokenizer')
local cmd = onmt.utils.ExtendedCmdLine.new('tokenize.lua')

tokenizer.declareOpts(cmd)

cmd:text('')
cmd:text('Other options')
cmd:text('')

cmd:option('-nparallel', 1, [[Number of parallel thread to run the tokenization]])
cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory]])

local opt = cmd:parse(arg)

local pool = threads.Threads(
  opt.nparallel,
  function()
    _G.separators = require('tools.utils.separators')
    _G.tokenizer = require('tools.utils.tokenizer')
    _G.BPE = require ('tools.utils.BPE')
    if opt.bpe_model ~= '' then
      _G.bpe = _G.BPE.new(opt)
    end
    if opt.normalize_cmd ~= '' then
      local N = require('tools.utils.normalizer')
      _G.normalizer = N.new(opt.normalize_cmd)
    end
  end
)
pool:specific(true)

local timer = torch.Timer()
local idx = 0

while true do
  local batches_input = {}
  local batches_output = {}
  local cidx = 0
  local line

  for i = 1, opt.nparallel do
    batches_input[i] = {}
    for _ = 1, opt.batchsize do
      line = io.read()
      if not line then break end
      cidx = cidx + 1
      table.insert(batches_input[i], line)
    end
    if not line then break end
  end

  if cidx == 0 then break end

  for i = 1, #batches_input do
    pool:addjob(
      i,
      function()
        local output = {}
        local inputs = batches_input[i]
        if _G.normalizer then
          inputs = _G.normalizer:normalize(inputs)
          if inputs == nil then
            inputs = batches_input[i]
            print("pb with normalizer - skipping "..#inputs)
          end
        end
        for b = 1,#inputs do
          local aline = inputs[b]
          local res
          local err

          local tokens

          res, err = pcall(function() tokens = _G.tokenizer.tokenize(opt, aline, _G.bpe) end)

          -- it can generate an exception if there are utf-8 issues in the text
          if not res then
            if string.find(err, "interrupted") then
              error("interrupted")
            else
              error("unicode error in line " .. idx+b*(i-1)*opt.batchsize .. ": " .. line .. '-' .. err)
            end
          end

          table.insert(output, table.concat(tokens, ' '))
        end
        return output
      end,
      function(output)
        batches_output[i] = output
      end
    )
  end

  pool:synchronize()

  -- output the tokenized and featurized string
  for i = 1, opt.nparallel do
    if not batches_output[i] then break end
    for b = 1, #batches_output[i] do
      io.write(batches_output[i][b] .. '\n')
      idx = idx + 1
    end
  end

  if not line then break end

end

io.stderr:write(string.format('Tokenization completed in %0.3f seconds - %d sentences\n', timer:time().real, idx))
