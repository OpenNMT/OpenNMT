local function zero(t)
  for i = 1, #t do
    t[i]:zero()
  end
end

local function append(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

return {
  zero = zero,
  append = append
}
