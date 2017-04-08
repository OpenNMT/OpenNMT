--[[ Thin layer over Element Research dpnn NCE Module
--]]

local dpnn, err = pcall(function() return require 'dpnn' end)

if dpnn then
  local NCEModule, parent = torch.class('onmt.NCEModule', 'nn.NCEModule')

  --[[

  --]]
  function NCEModule:__init(rnnSize, outputSizes, k, unigrams)
    parent.__init(self, rnnSize, outputSizes, k, unigrams)
    self.normalized = true
    self.LogSoftMax = true
  end

end
