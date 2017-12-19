require('torch')

local threads = require 'threads'

local cmd = torch.CmdLine()

local separators = require('tools.utils.separators')

cmd:text("")
cmd:text("**detokenize.lua**")
cmd:text("")

cmd:option('-case_feature', false, [[First feature is case feature]])
cmd:option('-joiner', separators.joiner_marker, [[the joiner marker]])
cmd:option('-nparallel', 1, [[Number of parallel thread to run the tokenization]])
cmd:option('-batchsize', 1000, [[Size of each parallel batch - you should not change except if low memory]])

local opt = cmd:parse(arg)

local pool = threads.Threads(
   opt.nparallel,
   function()
     _G.separators = require('tools.utils.separators')
     _G.tokenizer = require('tools.utils.tokenizer')
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
          local oline
          local res, err = pcall(function() oline = _G.tokenizer.detokenizeLine(opt, batches_input[i][b]) end)
          table.insert(output, oline)
          if not res then
            if string.find(err,"interrupted") then
              error("interrupted")
            else
              error("unicode error in line ".. idx+(i-1)*opt.batchsize+b ..": "..line..'-'..err)
            end
          end
        end
        return output
      end,
      function(output)
        batches_output[i] = output
      end)
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
end

io.stderr:write(string.format('Detokenization completed in %0.3f seconds - %d sentences\n',timer:time().real,idx))
