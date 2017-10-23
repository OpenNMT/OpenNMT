require('onmt.init')

local tester = ...

local fileReaderTest = torch.TestSuite()

function fileReaderTest.countLines()
  tester:eq(onmt.utils.FileReader.countLines('data/src-test.txt'), 2737)
  tester:eq(onmt.utils.FileReader.countLines('data/sigtraintrig.srcfeat', true), 1000)
end

return fileReaderTest
