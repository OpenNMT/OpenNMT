require('nngraph')

--[[
  Adding coverage model inside attention according to:
    - "Modeling Coverage for NMT" (https://arxiv.org/pdf/1601.04811.pdf)

  Coverage is a vector of size Bxlx1 or Bxlx10 for nn10
--]]


local GlobalAttentionCoverage, parent = torch.class('onmt.GlobalAttentionCoverage', 'onmt.Network')

local options = {
  {
    '-coverage_model', 'ling1',
    [[Coverage model type.]],
    {
      enum = { 'ling1' ,'ling2', 'nn1', 'nn10' }
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
  self.coverageSize = (opt.coverage_model == 'nn10' and 10) or 1
  parent.__init(self, self:_buildModel(opt, dim))
end

GlobalAttentionCoverage.hasCoverage = 1

function GlobalAttentionCoverage:_buildModel(opt, dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local ht0 = inputs[1] -- target context: batchLx dim
  local hs = inputs[2] -- source context: batchL x sourceTimesteps x dim
  local coverage = inputs[3] -- coverage vector: batchL x sourceTimesteps

  -- concatenate coverage to hs
  local hs_cov = nn.JoinTable(3)({hs, coverage})

  -- Get attention.
  local score_ht_hs
  local ht = nn.Linear(dim, dim+self.coverageSize, false)(ht0) -- batchL x dim
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
    coverage = nn.CAddTable()({coverage, nn.CDivTable()({attn, nn.AddConstant(0.00001)(phi)})})
  else
    -- apply GRU cell - coverage_t_i = gru(coverage_t-1, [attn_t_i, h_t, h_s_i])
    local ht2 = nn.Replicate(1,2)(ht0) -- batchL x 1 x dim
    local attn2 = nn.Replicate(1,3)(attn) -- batchL x sourceL x 1
    local ht_hs_attn = onmt.JoinReplicateTable(2,3)({ht2, hs, attn2})
    local gru = onmt.GRU.new(1, 2*dim+1, self.coverageSize)
    -- use bottle so that batch and timesteps are together
    coverage = onmt.Bottle(gru)({coverage, ht_hs_attn})
  end

  -- Apply attention to context.
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL
  local contextCombined = nn.MM()({attn, hs}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput, coverage})
end
