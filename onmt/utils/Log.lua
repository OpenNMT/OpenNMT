local function logJsonRecursive(obj)
  if type(obj) == 'string' then
    _G.logger:writeMsg('"' .. obj .. '"')
  elseif type(obj) == 'table' then
    local first = true

    _G.logger:writeMsg('{')

    for key, val in pairs(obj) do
      if not first then
        _G.logger:writeMsg(',')
      else
        first = false
      end
      _G.logger:writeMsg('"' .. key .. '":')
      logJsonRecursive(val)
    end

    _G.logger:writeMsg('}')
  else
    _G.logger:writeMsg(tostring(obj))
  end
end

--[[ Recursively outputs a Lua object to a JSON objects followed by a new line. ]]
local function logJson(obj)
  logJsonRecursive(obj)
  _G.logger:writeMsg('\n')
end

return {
  logJson = logJson
}
