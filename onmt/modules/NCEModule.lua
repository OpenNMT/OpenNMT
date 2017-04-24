--[[ Thin layer over Element Research dpnn NCE Module
--]]

local dpnn, _ = pcall(function() return require 'dpnn' end)

if dpnn then
  local NCEModule, parent = torch.class('onmt.NCEModule', 'nn.NCEModule')

  local options = {
    {
      '-nce_sample_size', 4096,
      [[Size of NCE sample.]],
      {
        valid = onmt.utils.ExtendedCmdLine.isUInt,
        structural = 1
      }
    },
    {
      '-nce_normalization', -1,
      [[Size of NCE normalization constant - if -1, will be estimated from first batch.]],
      {
        structural = 1
      }
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

  -- overload fastNoise function to avoid depedency with 'torchx'
  function NCEModule:fastNoise()
   -- we use alias to speedup multinomial sampling (see noiseSample method)
   self.unigrams:div(self.unigrams:sum())
   self.aliasmultinomial = onmt.AliasMultinomial(self.unigrams)
   self.aliasmultinomial.dpnn_parameters = {'J', 'q'}
end

end
