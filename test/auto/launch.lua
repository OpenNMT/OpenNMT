local yaml = require 'yaml'

local file = io.open("test/auto/config.yaml", "rb")
assert(file)
local content = file:read "*a"
file:close()

local config = yaml.load(content)

local params = config.params
local tests = config.tests
local data = config.data

local tmp_dir = params.tmp_dir

function log(message, level)
  level = level or 'INFO'
  local timeStamp = os.date('%x %X')
  local msgFormatted = string.format('[%s %s] %s', timeStamp, level, message)
  print (msgFormatted)
end

-- remove tmp directory
os.execute("rm -rf '"..tmp_dir.."'")
assert(os.execute("mkdir '"..tmp_dir.."'"))

idx=1

for _, test in pairs(tests) do
  local command=''
  local env = {
    TMP = tmp_dir..'/t_'..idx,
    PARAMS_PREPROCESS = '',
    PARAMS_TRAIN = ''
  }
  if test.params_train then
    env.PARAMS_TRAIN = test.params_train
  end
  if test.params_preprocess then
    env.PARAMS_PREPROCESS = test.params_preprocess
  end
  assert(os.execute("mkdir '"..env.TMP.."'"))
  for k,v in pairs(env) do
    command = command .. k .. "='" .. v .. "' "
  end
  command = command .. params.path_scripts .. '/' .. test.script
  command = command .. " > '"..env.TMP.."/stdout'"
  command = command .. " 2> '"..env.TMP.."/stderr'"

  log('test '..idx..' - LAUNCH '..test.name,'INFO')
  local f = io.open(env.TMP.."/cmdline", "w")
  f:write(command)
  f:close()

  local timer = torch.Timer()
  local res, info, val = os.execute (command)
  if res then
    log('test '..idx..' - COMPLETED ('..val..') in '..timer:time().real..' seconds', 'INFO')
  else
    log('test '..idx..' - FAILED ('..info..'/'..val..') in '..timer:time().real..' seconds', 'ERROR')
  end
  idx = idx + 1
end
