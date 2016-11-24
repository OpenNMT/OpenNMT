local function is_set(opt, name)
  return opt[name]:len() > 0
end

local function require_option(opt, name)
  if not is_set(opt, name) then
    error("option -" .. name .. " is required")
  end
end

local function require_options(opt, names)
  for i = 1, #names do
    require_option(opt, names[i])
  end
end

local function load_file(filename)
  local function strip(s)
    return s:gsub("^%s+",""):gsub("%s+$","")
  end

  local file = assert(io.open(filename, "r"))
  local opt = {}

  for line in file:lines() do
    if line:len() > 0 and string.sub(line, 1, 1) ~= '#' then
      local field = line:split('=')
      assert(#field == 2, 'badly formatted config file')
      local key = strip(field[1])
      local val = strip(field[2])
      opt[key] = val
    end
  end

  file:close()
  return opt
end

local function load_config(filename, opt)
  local config = load_file(filename)

  for key, val in pairs(config) do
    assert(opt[key] ~= nil, 'unkown option ' .. key)
    opt[key] = val
  end

  return opt
end

return {
  require_option = require_option,
  require_options = require_options,
  load_config = load_config
}
