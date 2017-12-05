require('onmt.init')

local tester = ...

local errorTest = torch.TestSuite()

function errorTest.noerror()
  tester:assertNoError(function() onmt.utils.Error.assert(true, "error") end)
end

return errorTest
