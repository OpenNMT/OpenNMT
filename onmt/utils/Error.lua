--[[ graceful handling of errors
--]]
local Error = torch.class('Error')

function Error.assert(condition, message, ...)
  if not condition then
    local trace = onmt.utils.String.split(debug.traceback(), "\n")
    local msgtrace = ""
    if #trace > 2 then
      msgtrace = " ("
      for i = 3, #trace-3 do
        if i ~= 3 then msgtrace = msgtrace .. " / " end
        msgtrace = msgtrace .. trace[i]:sub(2)
      end
      msgtrace = msgtrace .. ")"
    end
    if _G.logger then
      _G.logger:error(message..msgtrace.."\nABORTING...", ...)
    else
      print("ERROR:",string.format(message..msgtrace.."\nABORTING...", ...))
    end
    os.exit(0)
  end
  return condition
end

return Error
