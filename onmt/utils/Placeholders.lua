local function norm(t)
  if type(t) == "table" then
    local v = {}
    local vrep = {}
    for _, tokt in ipairs(t) do
      local vt, vtrep
      vt, vtrep = norm(tokt)
      table.insert(v, vt)
      table.insert(vrep, vtrep)
    end
    return v, vrep
  end
  if t:find('｟') then
    local p = t:find('｠')
    assert(p, 'invalid placeholder tag: '..t)
    local fields = onmt.utils.String.split(t, '：')
    return fields[1]..t:sub(p), fields[2] or fields[1]
  end
  return t
end

return {
  norm = norm
}
