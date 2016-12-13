local function is_set(opt, name)
  return opt[name]:len() > 0
end

--[[ Check that option `name` is set in `opt`. Throw an error if not set. ]]
local function require_option(opt, name)
  if not is_set(opt, name) then
    error("option -" .. name .. " is required")
  end
end

--[[ Make sure all options in `names` are set in `opt`. ]]
local function require_options(opt, names)
  for i = 1, #names do
    require_option(opt, names[i])
  end
end

--[[ Convert `val` string to its actual type (boolean, number or string). ]]
local function convert(val)
  if val == 'true' then
    return true
  elseif val == 'false' then
    return false
  else
    return tonumber(val) or val
  end
end

--[[ Return options set in the file `filename`. ]]
local function load_file(filename)
  local file = assert(io.open(filename, "r"))
  local opt = {}

  for line in file:lines() do
    -- Ignore empty or commented out lines.
    if line:len() > 0 and string.sub(line, 1, 1) ~= '#' then
      local field = line:split('=')
      assert(#field == 2, 'badly formatted config file')
      local key = utils.String.strip(field[1])
      local val = utils.String.strip(field[2])
      opt[key] = convert(val)
    end
  end

  file:close()
  return opt
end

--[[ Override `opt` with option values set in file `filename`. ]]
local function load_config(filename, opt)
  local config = load_file(filename)

  for key, val in pairs(config) do
    assert(opt[key] ~= nil, 'unkown option ' .. key)
    opt[key] = val
  end

  return opt
end

local function init(opt, required_options)
  if opt.config:len() > 0 then
    opt = load_config(opt.config, opt)
  end

  require_options(opt, required_options)

  if opt.seed then
    torch.manualSeed(opt.seed)
  end
end

return {
  init = init
}
