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
  {
    '-coverage_model', 'ling1',
    [[Coverage model type.]],
    {
      enum = { 'ling1' ,'ling2' }
    }
  }
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
  parent.__init(self, self:_buildModel(opt, dim))
end

GlobalAttentionCoverage.hasCoverage = 1

function GlobalAttentionCoverage:_buildModel(opt, dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local ht = inputs[1] -- target context: batchLx dim
  local hs = inputs[2] -- source context: batchL x sourceTimesteps x dim
  local coverage = inputs[3] -- coverage vector: batchL x sourceTimesteps

  -- concatenate coverage to hs
  local hs_cov = nn.JoinTable(3)({hs, nn.Replicate(1,3)(coverage)})

  -- Get attention.
  local score_ht_hs
  ht = nn.Linear(dim, dim+1, false)(ht) -- batchL x dim
  score_ht_hs = nn.MM()({hs_cov, nn.Replicate(1,3)(ht)}) -- batchL x sourceL x 1

  local attn = nn.Sum(3)(score_ht_hs) -- batchL x sourceL
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)

  -- update coverage
  if opt.coverage_model == 'ling1' then
    -- linguistic coverage model without fertility model: phi=1
    coverage = nn.CAddTable()({coverage, attn})
  elseif opt.coverage_model == 'ling2' then
    -- fertility model - phi=N.sigma(U.h_s)
    local phi = nn.Mul()(nn.Sigmoid()(nn.Bottle(nn.Linear(dim, 1))(hs)))
    coverage = nn.CAddTable()({coverage, nn.CDivTable()({attn, phi})})
  end
  -- apply GRU cell - coverage_t = f(coverage_t-1, attn_t, h_t, h_s)

  -- Apply attention to context.
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL
  local contextCombined = nn.MM()({attn, hs}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput, coverage})
end
