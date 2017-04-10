require('nngraph')

--[[ Global attention takes a matrix, a query vector and sum of previous attentions. It
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

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

--]]
local GlobalAttentionCoverage, parent = torch.class('onmt.GlobalAttentionCoverage', 'onmt.Network')

local options = {
}

function GlobalAttentionCoverage.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Global Attention Model')
end

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function GlobalAttentionCoverage:__init(opt, dim)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(opt, options)
  parent.__init(self, self:_buildModel(dim))
end

GlobalAttentionCoverage.hasCoverage = 1

function GlobalAttentionCoverage:_buildModel(dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local ht = inputs[1]
  local context = inputs[2] -- batchL x sourceTimesteps x dim
  local coverage = inputs[3] -- coverage vector

  -- Get attention.
  local score_ht_hs
  if global_attention ~= 'concat' then
    if global_attention == 'general' then
      ht = nn.Linear(dim, dim, false)(ht) -- batchL x dim
    end
    score_ht_hs = nn.MM()({context, nn.Replicate(1,3)(ht)}) -- batchL x sourceL x 1
  else
    local ht2 = nn.Replicate(1,2)(ht) -- batchL x 1 x dim
    local ht_hs = onmt.JoinReplicateTable(2,3)({ht2, context})
    local Wa_ht_hs = nn.Bottle(nn.Linear(dim*2, dim, false),2)(ht_hs)
    local tanh_Wa_ht_hs = nn.Tanh()(Wa_ht_hs)
    score_ht_hs = nn.Bottle(nn.Linear(dim,1),2)(tanh_Wa_ht_hs)
  end
  local attn = nn.Sum(3)(score_ht_hs) -- batchL x sourceL
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL

  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput, coverage})
end
