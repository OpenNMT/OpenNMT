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

return {
  require_option = require_option,
  require_options = require_options
}
