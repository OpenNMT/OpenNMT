--[[
--]]

local dpnn, err = pcall(function() return require 'dpnn' end)

if dpnn then
  local NCEModule, parent = torch.class('onmt.NCECriterion', 'nn.NCECriterion')

  --[[

  --]]
  function NCECriterion:__init()
    parent.__init(self)
  end

end
