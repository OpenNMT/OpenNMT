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

--]]
local GlobalAttention, parent = torch.class('onmt.GlobalAttention', 'nn.Container')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function GlobalAttention:__init(dim)
  parent.__init(self)
  self.net = self:_buildModel(dim)
  self:add(self.net)
end

function GlobalAttention:_buildModel(dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local target_t = nn.Linear(dim, dim, false)(inputs[1]) -- batch_l x dim
  local context = inputs[2] -- batch_l x source_timesteps x dim

  -- Get attention.
  local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
  attn = nn.Sum(3)(attn)
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  attn = nn.Replicate(1,2)(attn) -- batch_l x 1 x source_l

  -- Apply attention to context.
  local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x dim
  context_combined = nn.Sum(2)(context_combined) -- batch_l x dim
  context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x dim*2
  local context_output = nn.Tanh()(nn.Linear(dim*2, dim, false)(context_combined))

  return nn.gModule(inputs, {context_output})
end

function GlobalAttention:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function GlobalAttention:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function GlobalAttention:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end
