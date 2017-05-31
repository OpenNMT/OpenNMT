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

if opt.bpe_model ~= '' then
  local f = assert(io.open(opt.bpe_model, "r"))
  local options = {}
  for i in string.gmatch(f:read("*line"), "[^;]+") do table.insert(options, i) end
  if #options == 4 then opt.mode = options[4] end  -- overriding 'mode' from cmd by options from bpe_model for BPE compatibility
end

local pool = threads.Threads(
   opt.nparallel,
   function()
     _G.separators = require('tools.utils.separators')
     _G.tokenizer = require('tools.utils.tokenizer')
     _G.BPE = require ('tools.utils.BPE')
     if opt.bpe_model ~= '' then
       _G.bpe = _G.BPE.new(opt)
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
        for b = 1,#batches_input[i] do
          local res
          local err

          local tokens

          res, err = pcall(function() tokens = _G.tokenizer.tokenize(opt, batches_input[i][b], _G.bpe) end)

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
