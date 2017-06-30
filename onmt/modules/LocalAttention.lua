require('nngraph')

--[[Implement local attention from http://aclweb.org/anthology/D15-1166]]

local LocalAttention, parent = torch.class('onmt.LocalAttention', 'onmt.Attention')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
]]
function LocalAttention:__init(opt, dim)
  parent.__init(self, opt, self:_buildModel(dim, self.args.attention_type))
end

function LocalAttention.needSLen()
  return true
end

function LocalAttention:_buildModel(dim, attention_type)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local ht = inputs[1]
  local context = inputs[2] -- batchL x sourceTimesteps x dim
  local slen = inputs[3] -- batchL

  -- get pt first
  local Wp_ht = nn.Bottle(nn.Linear(dim, dim, false),2)(ht)
  local tanh_Wp_ht = nn.Tanh()(Wp_ht)
  local pt = nn.Sigmoid(nn.Bottle(nn.Linear(1, dim),2)(tanh_Wp_ht)) -- batchL
  pt = nn.CMul()(slen, pt)

  -- build context around pt
  local lcontext_mu = onmt.CenteredWindow(self.args.local_attention_span)(context, pt)
  local local_context = nn.SelectTable(1)(lcontext_mu)
  local mu = nn.SelectTable(2)(lcontext_mu)

  -- Get attention.
  local attn = self:buildAttention(local_context, ht, attention_type, dim)
  -- favor alignment points near p_t
  attn = nn.CMul()({attn, mu})

  -- Apply attention to context.
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x windowSize
  local contextCombined = nn.MM()({attn, local_context}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end
