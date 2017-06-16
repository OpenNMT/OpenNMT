
require('torch')
local path = require('pl.path')

require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('scorer.lua')

local scorers = {}
for n, _ in pairs(onmt.scorers) do
  table.insert(scorers, n)
end

cmd:option(
  '-scorer',
  'bleu',
  [[Scorer to use.]],
  {
    enum = scorers
  }
)
cmd:option(
  'rfilestem',
  '',
  [[Reads the reference(s) from `name` or `name`.0, `name`.1, ...]],
  {
    valid = onmt.utils.ExtendedCmdLine.nonEmpty
  }
)

local opt = cmd:parse(arg)

-- read the references
local references = {}

local function add_to_reference(filename)
  local file = io.open(filename)
  assert(file, "cannot open `" .. filename .. "`")
  local ref = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(ref, line)
  end
  assert(references==0 or #references[#references] == #ref, "error: all references don't have same line count")
  table.insert(references, ref)
end

local refid = 0

if path.exists(opt.rfilestem) then
  add_to_reference(opt.rfilestem)
end

while path.exists(opt.rfilestem .. '.' .. refid) do
  add_to_reference(opt.rfilestem .. '.' .. refid)
  refid = refid + 1
end

local hyp = {}
-- read from stdin
while true do
  local line = io.read()
  if not line then break end
  table.insert(hyp, line)
end

assert(#hyp==#references[1], "error: line count hyp/ref does not match")

