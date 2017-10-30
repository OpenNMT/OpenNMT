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
  if t:sub(1, string.len('｟')) == '｟' then
    assert(t:sub(-string.len('｠')) == '｠', 'invalid placeholder tag: '..t)
    local tcontent = t:sub(string.len('｟')+1, -string.len('｠')-1)
    local fields = onmt.utils.String.split(tcontent, '：')
    return '｟'..fields[1]..'｠', fields[2] or fields[1]
  end
  return t
end

return {
  norm = norm
}
