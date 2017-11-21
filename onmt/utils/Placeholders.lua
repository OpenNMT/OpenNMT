local function norm(t)
  if type(t) == "table" then
    local v = {}
    local vrep = {}
    for _, tokt in ipairs(t) do
      local vt, vtrep
      vt, vtrep = norm(tokt)
      table.insert(v, vt)
      if vtrep then
        vrep[vt] = vtrep
      end
    end
    return v, vrep
  end
  if t:sub(1, string.len('｟')) == '｟' then
    local p = t:find('｠')
    assert(p, 'invalid placeholder tag: '..t)
    local tcontent = t:sub(string.len('｟')+1, p-1)
    local fields = onmt.utils.String.split(tcontent, '：')
    local ph = '｟'..fields[1]..t:sub(p)
    return ph, fields[2] or ph
  end
  return t
end

return {
  norm = norm
}
