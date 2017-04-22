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

local runname = os.date('%y%m%d-%H%M%S')
local log_file_path = log_dir..'/'..runname..'.txt'
local log_file = assert(io.open(log_file_path, 'w'))
log_file:write('running autotest...\n')
log_file:close()

local handle = io.popen("git rev-parse --short HEAD")
local git_rev = handle:read("*a")
handle:close()

local function log(message, level)
  level = level or 'INFO'
  local timeStamp = os.date('%x %X')
  local msgFormatted = string.format('[%s %s] %s', timeStamp, level, message)
  log_file = assert(io.open(log_file_path, 'a'))
  log_file:write(msgFormatted..'\n')
  log_file:close()
end

local function setbadge(path, color, message)
  os.execute("wget --no-verbose 'https://img.shields.io/badge/autotest-"..message.."-"..color..".svg' -O '"..path.."'")
  os.execute("wget --no-verbose 'https://img.shields.io/badge/autotest-"..message.."-"..color..".svg' -O '"..log_dir.."/laststatus.svg'")
end

local idxCommand = 1
local function osExecute(command)
  log("EXECUTE ["..idxCommand.."]: ".. command, 'INFO')
  local timer = torch.Timer()
  local res, info, val = os.execute(command.." 2>&1 | perl -pe '$|=1;s/^/...... /' | tee -a '"..log_file_path.."'")
  if res then
    log('['..idxCommand..'] - COMPLETED ('..info..'/'..val..') in '..timer:time().real..' seconds', 'INFO')
  else
    log('['..idxCommand..'] - FAILED ('..info..'/'..val..') in '..timer:time().real..' seconds', 'ERROR')
    setbadge(log_file_path..'-status.svg', "red", "failed ("..git_rev..")")
    os.exit(0)
  end
  idxCommand = idxCommand+1
end

local function osAssert(test, message)
  if not test then
    log(message, 'ERROR')
    setbadge(log_file_path..'-status.svg', "red", "failed ("..git_rev..")")
    os.exit(0)
  end
end

setbadge(log_file_path..'-status.svg', "yellow", "running: prep ("..git_rev..")")
local alllog_file = assert(io.open(log_dir..'/alllog.html', 'a'))
alllog_file:write("<li><a href='./"..runname..".txt'>"..runname.."</a>: <img src='"..runname..".txt-status.svg'></li>\n")
alllog_file:close()


-- remove tmp directory
log('Prepare TMP_DIR', 'INFO')
-- this command can fail - just removing all tmp
os.execute("rm -rf '"..tmp_dir.."'")
osExecute("mkdir '"..tmp_dir.."'")

local data_path = {}

log('Download data', 'INFO')
os.execute("mkdir '"..download_dir.."'")
-- get associated data
for k, v in pairs(data) do
  local file, suffix = string.match(v, ".*/([^/]*)(.tgz)")
  if not os.execute("tar -tzf '"..download_dir.."/"..file..suffix.."' > /dev/null") then
    log(' * download '..v)
    osExecute("wget '"..v.."' -O '"..download_dir.."/"..file..suffix.."'")
    osExecute("tar -C '"..download_dir.."' -xzf '"..download_dir.."/"..file..suffix.."'")
  else
    log(' * already downloaded '..v)
  end
  -- change directory
  data_path[k]=download_dir.."/"..file
end

local idx=1

for _, test in pairs(tests) do
  setbadge(log_file_path..'-status.svg', "yellow", "running: "..idx..'/'..#tests.." on ("..git_rev..")")

  local command=''
  local env = {
    TMP = tmp_dir..'/t_'..idx
  }
  if test.data then
    osAssert(data_path[test.data], "missing data file: "..test.data)
    env.DATA = data_path[test.data]
  end
  if test.params then
    for k,v in pairs(test.params) do
      env[k] = v
    end
  end
  osExecute("mkdir '"..env.TMP.."'")
  for k,v in pairs(env) do
    command = command .. k .. "='" .. v .. "' "
  end
  command = command .. params.path_scripts .. '/' .. test.script

  log('test '..idx..' - LAUNCH '..test.name,'INFO')
  local f = io.open(env.TMP.."/cmdline", "w")
  f:write(command)
  f:close()

  osExecute (command)
  idx = idx + 1
end

setbadge(log_file_path..'-status.svg', "green", "pass ("..git_rev..")")
