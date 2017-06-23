local function calculate_one_dlratio(pred, ref)
  local predlen = string.len(pred)
  local reflen = string.len(ref)
  local matrix = {}

  -- save time if we can
  if (predlen == 0) then
    return reflen
  elseif (pred == ref) then
    return 0
  end

  -- create the predlen x reflen matrix
  for i = 0, predlen do
    matrix[i] = {}
    for j = 0, reflen do
      matrix[i][j] = 0
    end
  end

  -- initialize the matrix
  for i = 1, predlen do
    matrix[i][0] = i
  end
  for j = 1, reflen do
    matrix[0][j] = j
  end

  -- calculate Damerau-Levenshtein edit distance
  for i = 1, predlen, 1 do
    for j = 1, reflen, 1 do
      local predchar = string.byte(pred, i)
      local refchar = string.byte(ref, j)
      matrix[i][j] = math.min(
        matrix[i-1][j] + 1, -- Deletion
        matrix[i][j-1] + 1, -- Insertion
        matrix[i-1][j-1] + (predchar == refchar and 0 or 1) -- Substitution
      )
      if predchar == string.byte(ref, j-1) and refchar == string.byte(pred, i-1) then
        matrix[i][j] = math.min(matrix[i][j], matrix[i-2][j-2] + (predchar == refchar and 0 or 1)) -- Transposition
      end
    end
  end

  -- return DL edit dist
  return matrix[predlen][reflen]
end

local function calculate_dlratio(preds, refs)
  local reflensum = 0
  local totaldist = 0
  for x = 1, #preds do
    local pred = table.concat(preds[x], ' ')
    local ref = table.concat(refs[x], ' ')
    reflensum = reflensum + string.len(ref)
    totaldist = totaldist + calculate_one_dlratio(pred, ref)
  end
  return totaldist / reflensum
end

return calculate_dlratio