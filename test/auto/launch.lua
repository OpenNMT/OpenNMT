local yaml = require 'yaml'

local fconfig = io.open("test/auto/config.yaml", "rb")
assert(fconfig)
local content = fconfig:read "*a"
fconfig:close()

local config = yaml.load(content)

local params = config.params
local tests = config.tests
local data = config.data

local tmp_dir = params.tmp_dir
local log_dir = params.log_dir
local download_dir = params.download_dir

local function setbadge(path, color, message)
  assert(os.execute("wget 'https://img.shields.io/badge/autotest-"..message.."-"..color..".svg' -O '"..path.."'"), "cannot set badge'")
  assert(os.execute("wget 'https://img.shields.io/badge/autotest-"..message.."-"..color..".svg' -O '"..log_dir.."/laststatus.svg'"), "cannot set badge'")
end

local runname = os.date('%y%m%d-%H%M%S')
local log_file_path = log_dir..'/'..runname..'.txt'
local log_file = assert(io.open(log_file_path, 'w'))
setbadge(log_file_path..'-status.svg', "yellow", "running")
local alllog_file = assert(io.open(log_dir..'/alllog.html', 'a'))
alllog_file:write("<li><a href='./"..runname..".txt'>"..runname.."</a>: <img src='"..runname..".txt-status.svg'></li>\n")
alllog_file:close()

local function log(message, level)
  level = level or 'INFO'
  local timeStamp = os.date('%x %X')
  local msgFormatted = string.format('[%s %s] %s', timeStamp, level, message)
  print (msgFormatted)
end

-- remove tmp directory
log('Prepare TMP_DIR', 'INFO')
os.execute("rm -rf '"..tmp_dir.."'")
assert(os.execute("mkdir '"..tmp_dir.."'"))

local data_path = {}

log('Download data', 'INFO')
os.execute("mkdir '"..download_dir.."'")
-- get associated data
for k, v in pairs(data) do
  local file, suffix = string.match(v, ".*/([^/]*)(.tgz)")
  if not os.execute("tar -tzf '"..download_dir.."/"..file..suffix.."' > /dev/null") then
    log(' * download '..v)
    assert(os.execute("wget '"..v.."' -O '"..download_dir.."/"..file..suffix.."'"), "cannot not retrieve data file: '"..v.."'")
    assert(os.execute("tar -C '"..download_dir.."' -xzf '"..download_dir.."/"..file..suffix.."'", "cannot not untar '"..v.."'"))
  else
    log(' * already downloaded '..v)
  end
  -- change directory
  data_path[k]=download_dir.."/"..file
end

local idx=1

for _, test in pairs(tests) do
  local command=''
  local env = {
    TMP = tmp_dir..'/t_'..idx
  }
  if test.data then
    assert(data_path[test.data], "missing data file: "..test.data)
    env.DATA = data_path[test.data]
  end
  if test.params then
    for k,v in pairs(test.params) do
      env[k] = v
    end
  end
  assert(os.execute("mkdir '"..env.TMP.."'"))
  for k,v in pairs(env) do
    command = command .. k .. "='" .. v .. "' "
  end
  command = command .. params.path_scripts .. '/' .. test.script
  command = command .. " 2>&1 >> '"..log_file_path.."'"

  log('test '..idx..' - LAUNCH '..test.name,'INFO')
  local f = io.open(env.TMP.."/cmdline", "w")
  f:write(command)
  f:close()

  local timer = torch.Timer()
  local res, info, val = os.execute (command)
  if res then
    log('test '..idx..' - COMPLETED ('..info..'/'..val..') in '..timer:time().real..' seconds', 'INFO')
  else
    log('test '..idx..' - FAILED ('..info..'/'..val..') in '..timer:time().real..' seconds', 'ERROR')
    setbadge(log_file_path..'-status.svg', "red", "failed")
    os.exit(0)
  end
  idx = idx + 1
end

setbadge(log_file_path..'-status.svg', "green", "pass")
