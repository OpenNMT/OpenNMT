local function logJsonRecursive(obj)
  if type(obj) == 'string' then
    _G.logger:write('"' .. obj .. '"')
  elseif type(obj) == 'table' then
    local first = true

    _G.logger:write('{')

    for key, val in pairs(obj) do
      if not first then
        _G.logger:write(',')
      else
        first = false
      end
      _G.logger:write('"' .. key .. '":')
      logJsonRecursive(val)
    end

    _G.logger:write('}')
  else
    _G.logger:write(tostring(obj))
  end
end

--[[ Recursively outputs a Lua object to a JSON objects followed by a new line. ]]
local function logJson(obj)
  logJsonRecursive(obj)
  _G.logger:write('\n')
end

return {
  logJson = logJson
}
