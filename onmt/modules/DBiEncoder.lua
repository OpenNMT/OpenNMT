--[[ DBiEncoder is a deep bidirectional sequencer used for the source language.

It is a special case of a PDBiEncoder with pdbrnn_reduction = 1.

]]
local DBiEncoder, parent = torch.class('onmt.DBiEncoder', 'onmt.PDBiEncoder')

local options = {}

function DBiEncoder.declareOpts(cmd)
  onmt.BiEncoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end


--[[ Create a deep bidirectional encoder.

Parameters:

  * `args` - global arguments
  * `input` - input neural network.
]]
function DBiEncoder:__init(args, input)
  args.pdbrnn_reduction = 1
  args.pdbrnn_merge = 'sum' -- This value does not matter as there is no time reduction.
  parent.__init(self, args, input)
end

--[[ Return a new DBiEncoder using the serialized data `pretrained`. ]]
function DBiEncoder.load(pretrained)
  return parent.load(pretrained, 'onmt.DBiEncoder')
end

--[[ Return data to serialize. ]]
function DBiEncoder:serialize()
  return parent.serialize(self, 'DBiEncoder')
end
