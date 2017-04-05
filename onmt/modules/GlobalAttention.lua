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

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

  * dot: $$score(h_t,{\overline{h}}_s) = h_t^T{\overline{h}}_s$$
  * general: $$score(h_t,{\overline{h}}_s) = h_t^T W_a {\overline{h}}_s$$
  * concat: $$score(h_t,{\overline{h}}_s) = \nu_a^T tanh(W_a[h_t;{\overline{h}}_s])$$

--]]
local GlobalAttention, parent = torch.class('onmt.GlobalAttention', 'onmt.Network')

local options = {
  {'-global_attention', 'general',         [[Global attention model type.]],
        {enum={'general', 'dot', 'concat'}}}
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

function GlobalAttention:_buildModel(dim, global_attention)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local targetT
  if global_attention == 'general' then
    targetT = nn.Linear(dim, dim, false)(inputs[1]) -- batchL x dim
  else
    targetT = inputs[1]
  end
  local context = inputs[2] -- batchL x sourceTimesteps x dim

  -- Get attention.
  local attn
  if global_attention ~= 'concat' then
    attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
  else
    local extendedTarget = nn.Replicate(1,2)(targetT)
    attn = nn.Bottle(nn.Linear(dim,1),2)(
                nn.Tanh()(
                    nn.Bottle(nn.Linear(dim*2, dim, false),2)(
                        onmt.JoinReplicateTable(2,3)({extendedTarget, context}))))
  end
  attn = nn.Sum(3)(attn)
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL

  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end
