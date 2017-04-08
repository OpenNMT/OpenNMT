--[[
--]]

local dpnn, err = pcall(function() return require 'dpnn' end)

if dpnn then
  local NCECriterion, parent = torch.class('onmt.NCECriterion', 'nn.Criterion')

  --[[

  --]]
  function NCECriterion:__init(w)
    parent.__init(self)
    -- for training
    self.nce = nn.NCECriterion()
    self.nce.sizeAverage = false
    self.nll = nn.ClassNLLCriterion(w)
    self.nll.sizeAverage = false
    self.criterions = {self.nce, self.nll }
  end

  function NCECriterion:updateOutput(input, target)
    if type(input) == 'table' then
      return self.nce:updateOutput(input, target)
    else
      return self.nll:updateOutput(input, target)
    end
  end

  function NCECriterion:updateGradInput(input, target)
    return self.nce:updateGradInput(input, target)
  end

  function NCECriterion.normalize(gradInput, n)
    gradInput[1]:div(n)
    gradInput[2]:div(n)
  end

end
