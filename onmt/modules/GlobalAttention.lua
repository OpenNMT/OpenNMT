require('nngraph')

--[[ Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.

    H_1 H_2 H_3 ... H_n
     q   q   q       q
      |  |   |       |
       \ |   |      /
           .....
         \   |  /
             a

Constructs a unit mapping:
  $$(H_1 .. H_n, q) => (a)$$
  Where H is of `batch x n x dim` and q is of `batch x dim`.

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H); q] + b_2)$$.

  * dot: $$score(h_t,{\overline{h}}_s) = h_t^T{\overline{h}}_s$$
  * general: $$score(h_t,{\overline{h}}_s) = h_t^T W_a {\overline{h}}_s$$
  * concat: $$score(h_t,{\overline{h}}_s) = \nu_a^T tanh(W_a[h_t;{\overline{h}}_s])$$

--]]
local GlobalAttention, parent = torch.class('onmt.GlobalAttention', 'onmt.Network')

local options = {
  {
    '-global_attention', 'general',
    [[Global attention model type.]],
    {
      enum = {'general', 'dot', 'dot_scaled', 'concat'},
      structural = 0
    }
  },
  {
    '-multi_head_attention', 1,
    [[Number of attention head - should be a divisor of rnn_size]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt,
      depends = function (args) return args.rnn_size % args.multi_head_attention == 0, 'multi_head_attention should be a divisor of rnn_size' end
    }
  },
  {
    '-dropout_attention', 0,
    [[Dropout layer on attention]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0, 1)
    }
  }
}

function GlobalAttention.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Global Attention Model')
end

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function GlobalAttention:__init(opt, dim)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(opt, options)
  parent.__init(self, self:_buildModel(dim, self.args.global_attention))
end

function GlobalAttention:buildAttention(hs, ht, global_attention, dim)
  local score_ht_hs
  if global_attention ~= 'concat' then
    if global_attention == 'general' then
      ht = nn.Linear(dim, dim, false)(ht) -- batchL x dim
    end
    score_ht_hs = nn.MM()({hs, nn.Replicate(1,3)(ht)}) -- batchL x sourceL x 1
    if global_attention == 'scaled_dot' then
      score_ht_hs = nn.MulConstant(1/math.sqrt(dim))(score_ht_hs)
    end
  else
    local ht2 = nn.Replicate(1,2)(ht) -- batchL x 1 x dim
    local ht_hs = onmt.JoinReplicateTable(2,3)({ht2, hs})
    local Wa_ht_hs = nn.Bottle(nn.Linear(dim*2, dim, false),2)(ht_hs)
    local tanh_Wa_ht_hs = nn.Tanh()(Wa_ht_hs)
    score_ht_hs = nn.Bottle(nn.Linear(dim,1),2)(tanh_Wa_ht_hs)
  end

  local attn = nn.Sum(3)(score_ht_hs) -- batchL x sourceL
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)

  return nn.Dropout(self.args.dropout_attention)(attn)
end

function GlobalAttention:_buildModel(dim, global_attention)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local ht = inputs[1] -- batchL x dim
  local context = inputs[2] -- batchL x sourceL x dim

  local attn
  local contextCombined

  if self.args.multi_head_attention > 1 then
    -- linear layer on context and ht
    local sdim = dim / self.args.multi_head_attention


    local ht_l = nn.Linear(dim, dim, false)(ht)
    local context_l = nn.Bottle(nn.Linear(dim, dim, false), 2)(context)

    local contextCombined_l = {}

    -- split and build attentions
    for i = 1, self.args.multi_head_attention do
      local sht_l = nn.Narrow(2, (i-1)*sdim + 1, sdim)(ht_l)
      local scontext_l = nn.Narrow(3, (i-1)*sdim + 1, sdim)(context_l)
      local sattn = self:buildAttention(scontext_l, sht_l, global_attention, sdim)
      sattn = nn.Replicate(1,2)(sattn) -- batchL x 1 x sourceL
      table.insert(contextCombined_l, nn.Sum(2)(nn.MM()({sattn, scontext_l}))) -- batchL x sdim
    end

    -- concat
    contextCombined = nn.JoinTable(2)(contextCombined_l) -- batchL x { n x sdim } => batchL x dim

    -- last linear
    contextCombined = nn.Linear(dim, dim, false)(contextCombined)
  else
    attn = self:buildAttention(context, ht, global_attention, dim)
    attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL
    -- Apply attention to context.
    contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
    contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  end

  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end
