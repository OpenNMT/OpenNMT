require('onmt.init')

local tester = ...

local errorTest = torch.TestSuite()

function errorTest.error()
  local lvl = _G.logger.level
  _G.logger:setVisibleLevel('NOERROR')
  tester:assertError(function() onmt.utils.Error.assert(false, "error") end)
  _G.logger:setVisibleLevel(lvl)
end

function errorTest.noerror()
  tester:assertNoError(function() onmt.utils.Error.assert(true, "error") end)
end

return errorTest
