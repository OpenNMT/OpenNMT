--[[ ComplexEncoder combines Encoders



Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local ComplexEncoder, parent = torch.class('onmt.ComplexEncoder', 'nn.Container')

--[[ Create a complex encoder.

Parameters:

  * `encoders` - table of encoders
  * `mapping_functions` - function preparing input of each encoder based on global encoder input and output of each previous encoder
]]
function ComplexEncoder:__init(encoders)
  parent.__init(self)

  self.encoders = encoders

  self.args = self.args or {}

  for i=1,#encoders do
    self:add(self.encoders[i])
  end
end
