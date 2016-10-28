local function is_set(opt, name)
  return opt[name]:len() > 0
end

local function require_option(opt, name)
  if not is_set(opt, name) then
    io.stderr:write("Option -" .. name .. " is required.\n")
    return false
  end

  return true
end

local function require_options(opt, prog_name, names)
  local all_present = true

  for i = 1, #names do
    local is_present = require_option(opt, names[i])
    if not is_present then
      all_present = false
    end
  end

  if not all_present then
    io.stderr:write("\nSee 'th " .. prog_name .. " -h' for more information.\n")
  end

  return all_present
end

return {
  require_option = require_option,
  require_options = require_options
}
