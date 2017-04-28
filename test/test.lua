require('torch')
require('onmt.init')

local lfs = require('lfs')

local testDir = 'test/onmt'
local tester = torch.Tester()

local function registerTestFile(path, name)
  local tests = assert(loadfile(path))(tester)

  for k, v in pairs(tests.__tests) do
    tester:add(v, name .. '_' .. k)
  end
end

local function registerTestDirectory(dir)
  for file in lfs.dir(dir) do
    if file ~= "." and file ~= ".." then
      local name = string.gsub(file, '%..+$', '')
      registerTestFile(dir .. '/' .. file, name)
    end
  end
end

local function main()
  local nThreads = torch.getnumthreads()
  torch.setnumthreads(1)

  _G.logger = onmt.utils.Logger.new('', true)

  registerTestDirectory(testDir)

  if #arg > 0 then
    local testNames = {}
    for i = 1, #arg do
      table.insert(testNames, arg[i])
    end
    tester:run(testNames)
  else
    tester:run()
  end

  torch.setnumthreads(nThreads)

  if tester.errors[1] then
    os.exit(1)
  end
end

main()
