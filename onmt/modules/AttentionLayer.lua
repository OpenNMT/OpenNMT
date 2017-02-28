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

  This module returns the context vector ONLY. The intention is to combine the contextV with the hidden layer using a conditional RNN
--]]
local AttentionLayer, parent = torch.class('onmt.AttentionLayer', 'onmt.Network')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function AttentionLayer:__init(dim)
  parent.__init(self, self:_buildModel(dim))
end

function AttentionLayer:_buildModel(dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local targetT = nn.Linear(dim, dim, false)(inputs[1]) -- batchL x dim
  local context = inputs[2] -- batchL x sourceTimesteps x dim
  
  -- Get attention.
  local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
  attn = nn.Sum(3)(attn)
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL

  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  local contextVector = nn.Sum(2)(contextCombined) -- batchL x dim
  


  return nn.gModule(inputs, {contextVector})
end
