local function isSet(opt, name)
  return opt[name]:len() > 0
end

--[[ Check that option `name` is set in `opt`. Throw an error if not set. ]]
local function requireOption(opt, name)
  if not isSet(opt, name) then
    error("option -" .. name .. " is required")
  end
end

--[[ Make sure all options in `names` are set in `opt`. ]]
local function requireOptions(opt, names)
  for i = 1, #names do
    requireOption(opt, names[i])
  end
end

--[[ Convert `val` string to its actual type (boolean, number or string). ]]
local function convert(key, val, ref)
  local new

  if type(val) == type(ref) then
    new = val
  elseif type(ref) == 'boolean' then
    if val == 'true' then
      new = true
    elseif val == 'false' then
      new = false
    else
      error('option ' .. key .. ' expects a boolean value [true, false]')
    end
  elseif type(ref) == 'number' then
    new = tonumber(val)
    assert(new ~= nil, 'option ' .. key .. ' expects a number value')
  end

  return new
end

--[[ Override `opt` with option values set in file `filename`. ]]
local function loadConfig(filename, opt)
  local file = assert(io.open(filename, "r"))

  for line in file:lines() do
    -- Ignore empty or commented out lines.
    if line:len() > 0 and string.sub(line, 1, 1) ~= '#' then
      local field = line:split('=')
      assert(#field == 2, 'badly formatted config file')

      local key = onmt.utils.String.strip(field[1])
      local val = onmt.utils.String.strip(field[2])

      assert(opt[key] ~= nil, 'unkown option ' .. key)

      opt[key] = convert(key, val, opt[key])
    end
  end

  file:close()
  return opt
end

local function dump(opt, filename)
  local file = assert(io.open(filename, 'w'))

  for key, val in pairs(opt) do
    file:write(key .. ' = ' .. tostring(val) .. '\n')
  end

  file:close()
end

local function init(opt, requiredOptions)
  if opt.config:len() > 0 then
    opt = loadConfig(opt.config, opt)
  end

  requireOptions(opt, requiredOptions)

  if opt.seed then
    torch.manualSeed(opt.seed)
  end
end

return {
  dump = dump,
  init = init
}
