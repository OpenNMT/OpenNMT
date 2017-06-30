require('nngraph')

local GlobalAttention, parent = torch.class('onmt.GlobalAttention', 'onmt.Attention')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function GlobalAttention:__init(opt, dim)
  parent.__init(self, opt, self:_buildModel(dim, opt))
end

function GlobalAttention:_buildModel(dim, opt)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local ht = inputs[1]
  local context = inputs[2] -- batchL x sourceTimesteps x dim

  -- Get attention.
  local attn = self:buildAttention(context, ht, opt, dim)

  -- Apply attention to context.
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end
