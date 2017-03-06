require('torch')

local threads = require 'threads'
local separators = require('tools.utils.separators')

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
cmd:option('-bpe_model', '', [[Apply Byte Pair Encoding if the BPE model path is given. If the option is used, 'mode' will be overridden/set automatically if the BPE model specified by bpe_model is learnt using learn_bpe.lua]])
cmd:option('-EOT_marker', separators.EOT, [[Marker used to mark the end of token, use '</w>' for python models, otherwise default value ]])
cmd:option('-BOT_marker', separators.BOT, [[Marker used to mark the begining of token]])
cmd:option('-bpe_case_insensitive', false, [[Apply BPE internally in lowercase, but still output the truecase units. This option will be overridden/set automatically if the BPE model specified by bpe_model is learnt using learn_bpe.lua]])
cmd:option('-bpe_mode', 'suffix', [[Define the mode for bpe. This option will be overridden/set automatically if the BPE model specified by bpe_model is learnt using learn_bpe.lua. - 'prefix': Append '﹤' to the begining of each word to learn prefix-oriented pair statistics;
'suffix': Append '﹥' to the end of each word to learn suffix-oriented pair statistics, as in the original python script;
'both': suffix and prefix; 'none': no suffix nor prefix]])
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
