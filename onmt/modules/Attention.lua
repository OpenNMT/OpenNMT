local Attention, parent = torch.class('onmt.Attention', 'onmt.Network')

--[[ Attention model take a matrix and a query vector. It
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

Attention can be local in which case it applies on a windows centered around p_t, or global.

Attention types are:
  * dot: $$score(h_t,{\overline{h}}_s) = h_t^T{\overline{h}}_s$$
  * general: $$score(h_t,{\overline{h}}_s) = h_t^T W_a {\overline{h}}_s$$
  * concat: $$score(h_t,{\overline{h}}_s) = \nu_a^T tanh(W_a[h_t;{\overline{h}}_s])$$

]]

local options = {
  {
    '-attention', 'global',
    [[Attention model type.]],
    {
      enum = {'none', 'local', 'global'},
      structural = 0
    }
  },
  { '-attention_type', '',
    [[]],
    {
      deprecated = '-attention_model'
    }
  },
  {
    '-attention_model', 'general',
    [[Attention type.]],
    {
      enum = {'general', 'dot', 'dot_scaled', 'concat'},
      structural = 0
    }
  },
  {
    '-local_attention_span', 5,
    [[Half-width of the local attention window.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt,
      structural = 0
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

function Attention.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Attention Model')
end

function Attention:__init(opt, model)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(opt, options)
  parent.__init(self, model)
end

function Attention:buildScore(hs, ht, attention_type, dim)
  local score_ht_hs
  if attention_type ~= 'concat' then
    if attention_type == 'general' then
      ht = nn.Linear(dim, dim, false)(ht) -- batchL x dim
    end
    score_ht_hs = nn.MM()({hs, nn.Replicate(1,3)(ht)}) -- batchL x sourceL x 1
    if attention_type == 'dot_scaled' then
      score_ht_hs = nn.MulConstant(1/math.sqrt(dim))(score_ht_hs)
    end
  else
    local ht2 = nn.Replicate(1,2)(ht) -- batchL x 1 x dim
    local ht_hs = onmt.JoinReplicateTable(2,3)({ht2, hs})
    local Wa_ht_hs = nn.Bottle(nn.Linear(dim*2, dim, false),2)(ht_hs)
    local tanh_Wa_ht_hs = nn.Tanh()(Wa_ht_hs)
    score_ht_hs = nn.Bottle(nn.Linear(dim,1),2)(tanh_Wa_ht_hs)
  end

  return score_ht_hs
end

function Attention:buildAttention(hs, ht, attention_type, dim)
  local score_ht_hs = self:buildScore(hs, ht, attention_type, dim)

  local attn = nn.Squeeze(3)(score_ht_hs) -- batchL x sourceL
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)

  return nn.Dropout(self.args.dropout_attention)(attn)
end
