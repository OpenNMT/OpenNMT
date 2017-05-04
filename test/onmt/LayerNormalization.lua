require('onmt.init')

local tester = ...

local layerNormalizationTest = torch.TestSuite()

function layerNormalizationTest.forwardbackward()
  local t = torch.Tensor{0.5,0.7,0.2,-6}
  local m=onmt.LayerNormalization(4)

  tester:eq(m:forward(t),torch.Tensor{0.588,0.659,0.481,-1.729},0.01)

  m=onmt.LayerNormalization(4, -1)

  tester:eq(m:forward(t),torch.Tensor{-0.4119,-0.3407,-0.5189,-2.7286},0.01)
end

return layerNormalizationTest
