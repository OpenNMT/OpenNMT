--[[ Thin layer over Element Research dpnn NCE Module
--]]

local dpnn, _ = pcall(function() return require 'dpnn' end)

if dpnn then
  local NCEModule, parent = torch.class('onmt.NCEModule', 'nn.NCEModule')

  local options = {
    {
      '-nce_sample_size', 25,
      [[Size of NCE sample.]]
    }
  }

  function NCEModule.declareOpts(cmd)
    cmd:setCmdLineOptions(options, 'Global Attention Model')
  end

  --[[

  --]]
  function NCEModule:__init(opt, rnnSize, outputSizes, unigrams)
    parent.__init(self, rnnSize, outputSizes, opt.nce_sample_size, unigrams)
    self.normalized = true
    self.logsoftmax = true

    -- fix bug in NCEModule:updateGradInput module
    -- type of gradInput for "target" (long tensor) is a float tensor which is useless but breaks other modules
    local updateGradInput = self.updateGradInput
    self.updateGradInput = function(modself, inputTable, gradOutput)
      local gradInput = updateGradInput(modself, inputTable, gradOutput)
      gradInput[2] = inputTable[2]
      return gradInput
    end
  end

end
