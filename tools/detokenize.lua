require('torch')

local tokenizer = require ('tools.utils.tokenizer')

local cmd = torch.CmdLine()

local separators = require('tools.utils.separators')

cmd:text("")
cmd:text("**detokenize.lua**")
cmd:text("")

cmd:option('-case_feature', false, [[First feature is case feature]])
cmd:option('-joiner', separators.joiner_marker, [[Character used to annotate joiners]])

local opt = cmd:parse(arg)

local timer = torch.Timer()
local idx = 1
for line in io.lines() do
  local res, err = pcall(function() io.write(tokenizer.detokenize(line, opt) .. '\n') end)
  if not res then
    if string.find(err,"interrupted") then
      error("interrupted")
    else
      error("unicode error in line "..idx..": "..line..'-'..err)
    end
  end
  idx = idx + 1
end

io.stderr:write(string.format('Detokenization completed in %0.3f seconds - %d sentences\n',timer:time().real,idx-1))
