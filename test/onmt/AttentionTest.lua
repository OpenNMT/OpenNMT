require('onmt.init')

local tester = ...

local attentionTest = torch.TestSuite()

local function checkDim(mod, dim, slen)
  local ht = torch.Tensor(4, dim):uniform(-0.1, 0.1)
  local context = torch.Tensor(4, 10, dim):uniform(-0.1, 0.1)

  local attnInput = {ht, context}
  table.insert(attnInput, slen)

  local output = mod:forward(attnInput)

  tester:eq(torch.isTensor(output), true)
  tester:eq(output:size(), torch.LongStorage({4, dim}))
end

function attentionTest.global_general()
  local attn = onmt.GlobalAttention({ attention_type = 'general', attention_dropout = 0 }, 50)
  checkDim(attn, 50)

  local softmaxAttn
  attn:apply(function (layer)
    if layer.name == 'softmaxAttn' then
      softmaxAttn = layer
    end
  end)

  tester:assert(softmaxAttn ~= nil)
  tester:eq(softmaxAttn.output:size(), torch.LongStorage({4, 10}))

end

function attentionTest.local_general()
  local attn = onmt.LocalAttention({ attention_type = 'general', attention_dropout = 0, local_attention_span = 3 }, 50)
  checkDim(attn, 50, torch.Tensor{10, 10, 10, 10})

  local softmaxAttn
  attn:apply(function (layer)
    if layer.name == 'softmaxAttn' then
      softmaxAttn = layer
    end
  end)

  tester:assert(softmaxAttn ~= nil)
  tester:eq(softmaxAttn.output:size(), torch.LongStorage({4, 7}))

end

function attentionTest.global_dot()
  local attn = onmt.GlobalAttention({ attention_type = 'dot', attention_dropout = 0.1 }, 50)
  checkDim(attn, 50)
end

function attentionTest.global_concat()
  local attn = onmt.GlobalAttention({ attention_type = 'concat', attention_dropout = 0.2 }, 50)
  checkDim(attn, 50)
end

function attentionTest.none()
  local attn = onmt.NoAttention(nil, 50)
  checkDim(attn, 50)
end

return attentionTest
