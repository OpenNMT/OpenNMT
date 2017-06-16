local function get_ngrams(ngrams, s, n)
  for i = 1, #s do
    for j = i, math.min(i+n-1, #s) do
      local ngram = table.concat(s, ' ', i, j)
      local l = j-i+1 -- keep track of ngram length
      if ngrams[ngram] == nil then
        ngrams[ngram] = {1, l}
      else
        ngrams[ngram][1] = ngrams[ngram][1] + 1
      end
    end
  end
  return ngrams
end

local function my_log(v)
  if v == 0 then
    return -9999999999
  end
  return math.log(v)
end

local function calculate_bleu_excludeminmax(cand, refs, N, min, max)
  local length_translation = 0
  local length_reference = 0
  local total = {}
  local correct = {}

  local actual_size = 0
  for i = 1, #cand do
    if not min or i < min or i > max then
      local cand_length = #cand[i]
      local closest_diff = 9999
      local closest_length = 9999

      local ref_ngram = {}

      for _, ref in ipairs(refs) do
        local ref_length = #ref[i]
        local diff = math.abs(cand_length-ref_length)
        if diff < closest_diff then
          closest_diff = diff
          closest_length = ref_length
        elseif diff == closest_diff then
          if ref_length < closest_length then
            closest_length = ref_length
          end
        end

        local ref_ngrams_n = {}
        get_ngrams(ref_ngrams_n, ref[i], N)

        for k, v in pairs(ref_ngrams_n) do
          if not ref_ngram[k] or ref_ngram[k][1] < v[1] then
            ref_ngram[k] = v
          end
        end
      end

      length_translation = length_translation + cand_length
      length_reference = length_reference+ closest_length

      local t_gram = {}
      get_ngrams(t_gram, cand[i], N)

      for k,v in pairs(t_gram) do
        local n = v[2]
        total[n] = (total[n] or 0) + v[1]
        if ref_ngram[k] then
          correct[n] = (correct[n] or 0) + math.min(v[1], ref_ngram[k][1])
        end
      end
      actual_size = actual_size + 1
    end
  end

  local nbleu = {}

  for n = 1, N do
    if total[n] then
      nbleu[n] = (correct[n] or 0) / total[n]
    else
      nbleu[n] = 0
    end
  end

  if length_reference == 0 then
    return 0, nbleu, -1, -1, length_translation, length_reference
  end

  local brevity_penalty = 1;

  if length_translation < length_reference then
    brevity_penalty = math.exp(1-length_reference/length_translation)
  end

  local bleu = 0

  for n = 1, N do
    bleu = bleu + my_log(nbleu[n])
  end

  bleu = brevity_penalty * math.exp(bleu/N)

  return bleu, nbleu, brevity_penalty, length_translation / length_reference, length_translation, length_reference
end


local function calculate_bleu(cand, refs, sample, N)
  N = N or 4
  sample = sample or 1

  local bleu, nbleu, bp, lratio, ltrans, lref = calculate_bleu_excludeminmax(cand, refs, N)

  local margin = 0

  if sample > 1 then
    local s = #cand
    for k = 1, sample do
      local sbleu = select(1, calculate_bleu_excludeminmax(cand, refs, N, (k-1)*s/sample, k*s/sample))
      if math.abs(sbleu-bleu) > margin then
        margin = math.abs(sbleu-bleu)
      end
    end
  end

  local vs = { bleu*100, margin*100 }
  local format = "BLEU = %.2f +/- %.2f, "
  for n = 1, N do
    if n > 1 then format = format .. '/' end
    format = format .. "%.1f"
    table.insert(vs, nbleu[n]*100)
  end
  table.insert(vs, bp)
  table.insert(vs, lratio)
  table.insert(vs, ltrans)
  table.insert(vs, lref)

  format = format .. " (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)"

  table.insert(vs, 1, format)

  return bleu, string.format(table.unpack(vs))
end

return calculate_bleu
