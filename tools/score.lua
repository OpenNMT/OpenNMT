
require('torch')
local path = require('pl.path')

require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('scorer.lua')

local options = {
  {
    '-scorer', 'bleu',
    [[Scorer to use.]],
    {
      enum = onmt.scorers.list
    }
  },
  {
    'rfilestem',
    'string',
    [[Reads the reference(s) from `name` or `name0`, `name1`, ...]],
    {
      valid = onmt.utils.ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-sample',
    10,
    [[Number of sample for estimation of error margin.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  },
  {
    '-order',
    4,
    [[Number of sample for estimation of error margin.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt()
    }
  }
}

cmd:setCmdLineOptions(options)

onmt.utils.Logger.declareOpts(cmd)

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  -- read the references
  local references = {}

  local function add_to_reference(filename)
    local file = io.open(filename)
    assert(file, "cannot open `" .. filename .. "`")
    local ref = {}
    while true do
      local line = file:read()
      if not line then break end
      local sent = {}
      for word in line:gmatch'([^%s]+)' do
        table.insert(sent, word)
      end
      table.insert(ref, sent)
    end
    assert(#references==0 or #references[#references] == #ref, "ERROR: all references do not have same line count")
    table.insert(references, ref)
  end

  local refid = 0

  if path.exists(opt.rfilestem) then
    add_to_reference(opt.rfilestem)
  end

  if onmt.scorers.multi[opt.scorer] then
    while path.exists(opt.rfilestem .. refid) do
      add_to_reference(opt.rfilestem .. refid)
      refid = refid + 1
    end
  end

  local hyp = {}
  -- read from stdin
  while true do
    local line = io.read()
    if not line then break end
    local sent = {}
    for word in line:gmatch'([^%s]+)' do
      table.insert(sent, word)
    end
    table.insert(hyp, sent)
  end

  assert(#hyp==#references[1], "ERROR: line count hyp/ref does not match")

  if not onmt.scorers.multi[opt.scorer] then
    references = references[1]
  end

  local score, format = onmt.scorers[opt.scorer](hyp, references, opt.sample, opt.order)
  print(format or score)
end

main()
