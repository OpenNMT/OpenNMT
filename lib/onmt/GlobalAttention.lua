require 'nngraph'

local GlobalAttention, parent = torch.class('onmt.GlobalAttention', 'nn.Module')

function GlobalAttention:__init(dim)
  parent.__init(self)
  self.net = self:_buildModel(dim)
end

--[[ Create an nngraph attention unit of size `dim`.

Returns: An nngraph unit mapping:
  ${(H_1 .. H_n, q) => (a)}$
  Where H is of `batch x n x dim` and q is of `batch x dim`.

  The full function is  ${\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)}$.

TODO:
  * allow different query and context sizes.
--]]
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
  return self.net:updateGradInput(input, gradOutput)
end

function GlobalAttention:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end

function GlobalAttention:parameters()
  return self.net:parameters()
end
