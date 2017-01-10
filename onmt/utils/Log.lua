local write
if _G.logger then
  write = function (...) return _G.logger:write(...) end
else
  write = io.write
end

local function logJsonRecursive(obj)
  if type(obj) == 'string' then
    write('"' .. obj .. '"')
  elseif type(obj) == 'table' then
    local first = true

    write('{')

    for key, val in pairs(obj) do
      if not first then
        write(',')
      else
        first = false
      end
      write('"' .. key .. '":')
      logJsonRecursive(val)
    end

    write('}')
  else
    write(tostring(obj))
  end
end

--[[ Recursively outputs a Lua object to a JSON objects followed by a new line. ]]
local function logJson(obj)
  logJsonRecursive(obj)
  write('\n')
end

return {
  logJson = logJson
}
