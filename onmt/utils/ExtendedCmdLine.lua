local onmtCmdLine, parent = torch.class('extendedCmdLine', 'torch.CmdLine')

function extendedCmdLine:__init()
  parent.__init(self)
end

function extendedCmdLine:option(key, default, help, _meta_)
  parent.option(self, key, default, help)
  self.options[key].meta = _meta_
  table.insert(self.helplines, self.options[key])
end

function extendedCmdLine:parse(arg)
  params = parent.parse(self, arg)
  for k,v in pairs(params) do
    local meta = self.options['-'..k].meta
    if meta and meta.valid then
      if not meta.valid(v) then
        self:error("option '"..k.."' value is not valid")
      end
    end
  end
end

function extendedCmdLine.integer(minValue, maxValue)
  return function(v)
    return (math.floor(v) == v and
            (not minValue or v >= minValue) and
            (not maxValue or v <= maxValue))
  end
end
