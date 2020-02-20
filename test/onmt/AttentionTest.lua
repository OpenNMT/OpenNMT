require('onmt.init')

local tester = ...

local attentionTest = torch.TestSuite()

local function checkDim(mod, dim)
  local ht = torch.Tensor(4, dim):uniform(-0.1, 0.1)
  local context = torch.Tensor(4, 10, dim):uniform(-0.1, 0.1)

  local output = mod:forward({ht, context})

  tester:eq(torch.isTensor(output), true)
  tester:eq(output:size(), torch.LongStorage({4, dim}))
end

function attentionTest.global_general()
  local attn = onmt.GlobalAttention({ global_attention = 'general' }, 50)
  checkDim(attn, 50)
end

function attentionTest.global_general_multihead_dropout()
  local attn = onmt.GlobalAttention({ global_attention = 'general', multi_head_attention = 2, dropout_attention = 0.2 }, 50)
  checkDim(attn, 50)
end

function attentionTest.global_dot()
  local attn = onmt.GlobalAttention({ global_attention = 'dot' }, 50)
  checkDim(attn, 50)
end

function attentionTest.global_dot_scaled()
  local attn = onmt.GlobalAttention({ global_attention = 'dot_scaled' }, 50)
  checkDim(attn, 50)
end

function attentionTest.global_concat()
  local attn = onmt.GlobalAttention({ global_attention = 'concat' }, 50)
  checkDim(attn, 50)
end

function attentionTest.none()
  local attn = onmt.NoAttention(nil, 50)
  checkDim(attn, 50)
end

return attentionTest
