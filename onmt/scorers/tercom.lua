-- implementation based on tercom v6 perl

----------------------- DEFAULT PARAMETERS -----------------------

-- standard costs
local MATCH_COST = 0
local INSERT_COST = 1
local DELETE_COST = 1
local SUB_COST = 1
local SHIFT_COST = 1

-- Super high value used to mark an impossible path
local INF = 99999999999

-- Maximum Length Sequence to Shift
-- Set to 0 to turn on shifting
local MAX_SHIFT_SIZE = 10

-- Maximum Distance To Shift
local MAX_SHIFT_DIST = 50

----------------------- ACTUAL CODE -----------------------

local function _min_edit_dist(i, j, hw, rw, mat, pat, full)
  -- recursively calculate the min edit path

  -- finalized exploration
  if i == 0 and j == 0 then return 0 end
  -- should never happen
  if i < 0 or j < 0 then return INF end

  -- extra-words horizontally or vertically
  if i == 0 then return j * DELETE_COST end
  if j == 0 then return i * INSERT_COST end

  -- already calculated
  if mat[i][j] then return mat[i][j] end

  -- diagonal cost
  local dia_cost = _min_edit_dist(i-1, j-1, hw, rw, mat, pat, full)

  local mcost = INF
  local scost = INF

  if hw[i] == rw[j] then
    mcost = MATCH_COST + dia_cost
  else
    scost = SUB_COST + dia_cost
  end

  local ip_cost = _min_edit_dist(i-1, j, hw, rw, mat, pat, full)
  local dp_cost = _min_edit_dist(i, j-1, hw, rw, mat, pat, full)

  local icost = INSERT_COST + ip_cost
  local dcost = DELETE_COST + dp_cost

  if mcost <= icost and mcost <= dcost and mcost <= scost then
    -- Match is best
    mat[i][j] = mcost
    pat[i][j] = " "
  elseif full and scost <= icost and scost <= dcost then
    -- local function is best
    mat[i][j] = scost
    pat[i][j] = "S"
  elseif icost <= dcost then
    -- Insert is best
    mat[i][j] = icost
    pat[i][j] = "I"
  else
    -- Deletion is best
    mat[i][j] = dcost
    pat[i][j] = "D"
  end

  return mat[i][j]
end

local function backtrace_path(pat, i, j)
  -- backtrace the min-edit-path
  local path = {}
  while i >= 1 or j >= 1 do
    if i < 1 then
      table.insert(path, 1, "D")
      j = j - 1
    elseif j < 1 then
      table.insert(path, 1, "I")
      i = i - 1
    else
      table.insert(path, 1, pat[i][j])
      if pat[i][j] == " " or pat[i][j] == "S" then
        i = i - 1
        j = j - 1
      elseif pat[i][j] == "I" then
        i = i - 1
      else
        j = j - 1
      end
    end
  end

  return path
end

local function min_edit_dist_arr(hw, rw, full)
  -- calculate the min-edit-dist on array of words
  local mat = {}
  local pat = {}

  for _ = 1, #hw do
    table.insert(mat, {})
    table.insert(pat, {})
  end

  _min_edit_dist(#hw, #rw, hw, rw, mat, pat, full)

  local score = mat[#hw][#rw]
  local path = backtrace_path(pat, #hw, #rw)

  return score, path
end

local function perform_shift(hwords, startpos, endpos, moveto)
  -- perform a shift on a string of words
  local range = {}
  if moveto == 0 then
    for j = startpos, endpos do
      table.insert(range, hwords[j])
    end
    for j = 0, startpos - 1 do
      table.insert(range, hwords[j])
    end
    for j = endpos + 1, #hwords do
      table.insert(range, hwords[j])
    end
  elseif moveto < startpos then
    for j = 1, moveto do
      table.insert(range, hwords[j])
    end
    for j = startpos, endpos do
      table.insert(range, hwords[j])
    end
    for j = moveto + 1, startpos - 1 do
      table.insert(range, hwords[j])
    end
    for j = endpos + 1, #hwords do
      table.insert(range, hwords[j])
    end
  elseif moveto > endpos then
    for j = 1, startpos-1 do
      table.insert(range, hwords[j])
    end
    for j = endpos+1, moveto do
      table.insert(range, hwords[j])
    end
    for j = startpos, endpos do
      table.insert(range, hwords[j])
    end
    for j = moveto + 1, #hwords do
      table.insert(range, hwords[j])
    end
  else
    -- we are moving inside of ourselves
    for j = 1, startpos-1 do
      table.insert(range, hwords[j])
    end
    for j = endpos+1, endpos + moveto - startpos do
      table.insert(range, hwords[j])
    end
    for j = startpos, endpos do
      table.insert(range, hwords[j])
    end
    for j = endpos + moveto - startpos + 1, #hwords do
      table.insert(range, hwords[j])
    end
  end

  return range
end

local function gather_all_poss_shifts(hwords, rloc, ralign, herr, rerr, min_size)
  -- find all possible shifts to search through
  local poss = {}
  local max_poss = -1

  -- return an array (@poss), indexed by len of shift
  -- each entry is (startpos, end, moveto)
  for startpos = 1, #hwords do
    if rloc[hwords[startpos]] then
      local ok = false

      for _, moveto in ipairs(rloc[hwords[startpos]]) do
          ok = ok or (startpos ~= ralign[moveto] and
                      ralign[moveto] - startpos <= MAX_SHIFT_DIST and
                      startpos - ralign[moveto]-1 <= MAX_SHIFT_DIST)
      end

      local endpos = startpos + min_size - 1
      while ok and endpos <= #hwords and endpos < startpos + MAX_SHIFT_SIZE do
        local cand_range = {}
        for j = startpos, endpos do
          table.insert(cand_range, hwords[j])
        end
        local cand = table.concat(cand_range, " ")

        ok = false
        if rloc[cand] then
          local any_herr = false
          local i = 0
          while i <= endpos - startpos and not any_herr do
            any_herr = herr[startpos+i]
            i = i + 1
          end
          if not any_herr then
            ok = true
          else
            -- consider moving startpos..end
            for _, moveto in ipairs(rloc[cand]) do
              if ralign[moveto] ~= startpos and
                 (ralign[moveto] < startpos or ralign[moveto] > endpos) and
                 ralign[moveto] - startpos <= MAX_SHIFT_DIST and
                 startpos - ralign[moveto] - 1 <= MAX_SHIFT_DIST then
                ok = true

                -- check to see if there are any errors in either string
                -- (only move if this is the case!)
                local any_rerr = false
                i = 0
                while i <= endpos - startpos and not any_rerr do
                  any_rerr = rerr[moveto+i]
                  i = i + 1
                end

                if any_rerr then
                  for roff = 0, endpos - startpos do
                    if startpos ~= ralign[moveto+roff] and
                       (roff == 0 or ralign[moveto+roff] ~= ralign[moveto]) then
                      if not poss[endpos-startpos] then
                        poss[endpos - startpos] = {}
                      end
                      if endpos - startpos > max_poss then
                        max_poss = endpos - startpos
                      end
                      table.insert(poss[endpos - startpos], {startpos, endpos, moveto+roff})
                    end
                  end
                end
              end
            end
          end
        end
        endpos = endpos + 1
      end
    end
  end
  return poss, max_poss
end

local function build_word_matches(harr, rarr)
  -- take in two arrays of words
  -- build a hash mapping each valid subseq of the ref to its location
  -- this is a utility func for calculating shifts
  local rloc = {}

  -- do a quick pass to check to see which words occur in both strings
  local hwhash = {}
  local cor_hash = {}
  for _, w in ipairs(harr) do
    hwhash[w] = 1
  end
  for _, w in ipairs(rarr) do
    cor_hash[w] = cor_hash[w] or hwhash[w]
  end
  -- build a hash of all the reference sequences
  for startpos = 1, #rarr do
    if cor_hash[rarr[startpos]] then
      local endpos = startpos
      local last = false
      while not last and endpos <= math.min(#rarr, startpos + MAX_SHIFT_SIZE) do
        if cor_hash[rarr[endpos]] then
          -- add sequence start...end to hash
          local range = {}
          for k = startpos, endpos do
            table.insert(range, rarr[k])
          end
          local topush = table.concat(range, " ")
          if not rloc[topush] then
            rloc[topush] = {}
          end
          table.insert(rloc[topush], startpos)
        else
          last = true
        end
        endpos = endpos + 1
      end
    end
  end
  return rloc
end

local function calc_best_shift(hyp, ref, rloc, curerr, path_vals)
  -- one greedy step in finding the shift
  -- find the best one at this point and return it

  local cur_best_score = curerr
  local cur_best_shift_cost = 0
  local cur_best_path = ""
  local cur_best_hyp = ""
  local cur_best_start = 0
  local cur_best_end = 0
  local cur_best_dest = 0

  local ralign = {}

  -- boolean. true if words[i] is an error
  local herr = {}
  local rerr = {}

  local hpos = 0
  for _, sym in ipairs(path_vals) do
    if sym == " " then
      hpos = hpos + 1
      table.insert(herr, false)
      table.insert(rerr, false)
      table.insert(ralign, hpos)
    elseif sym == "S" then
      hpos = hpos + 1
      table.insert(herr, true)
      table.insert(rerr, true)
      table.insert(ralign, hpos)
    elseif sym == "I" then
      hpos = hpos + 1
      table.insert(herr, true)
    elseif sym == "D" then
      table.insert(rerr, true)
      table.insert(ralign, hpos)
    end
  end

  -- Have we found any good shift yet?
  local anygain = false

  local poss_shifts, max_pos_shifts = gather_all_poss_shifts(hyp, rloc, ralign, herr, rerr, 1)

  local stop = false
  local i = max_pos_shifts
  while i >= 0 and not stop do
    local curfix = curerr - (cur_best_shift_cost + cur_best_score)
    local maxfix = 2 * (1 + i) - SHIFT_COST

    stop = curfix > maxfix or (cur_best_shift_cost ~= 0 and curfix == maxfix)

    if not stop then
      local work_start = -1
      local work_end = -1

      local j = 1
      while not stop and poss_shifts[i] and j <= #poss_shifts[i] do
        local s = poss_shifts[i][j]
        curfix = curerr - (cur_best_shift_cost + cur_best_score)
        maxfix = (2 * (1 + i)) - SHIFT_COST

        stop = curfix > maxfix or (cur_best_shift_cost ~= 0 and curfix == maxfix)

        if not stop then
          local startpos, endpos, moveto = table.unpack(s)
          if work_start == -1 then
            work_start, work_end = startpos, endpos
          elseif work_start ~= startpos and work_end ~= endpos then
            if not anygain then
              work_start, work_end = startpos, endpos
            end
          end

          local shifted_str = perform_shift(hyp, startpos, endpos, ralign[moveto])
          local try_score, try_path = min_edit_dist_arr(shifted_str, ref, 1)

          local gain = (cur_best_score + cur_best_shift_cost) - (try_score + SHIFT_COST)
          if gain > 0 or (cur_best_shift_cost == 0 and gain == 0) then
            anygain = true
            cur_best_score = try_score
            cur_best_shift_cost = SHIFT_COST
            cur_best_path = try_path
            cur_best_hyp = shifted_str
            cur_best_start = startpos
            cur_best_end = endpos
            cur_best_dest = ralign[moveto]
          end
        end
        j = j + 1
      end
    end
    i = i - 1
  end

  return cur_best_hyp, cur_best_score, cur_best_path, cur_best_start, cur_best_end, cur_best_dest
end

local function calc_shifts(cand, ref)
  local rloc = build_word_matches(cand, ref)
  local med_score, med_path = min_edit_dist_arr(cand, ref, 1)

  local edits = 0
  local cur = cand
  local all_shifts = {}

  while 1 do
    local new_hyp, new_score, new_path, sstart, send, sdest =
      calc_best_shift(cur, ref, rloc, med_score, med_path)
    if new_hyp == '' then
      break
    end

    table.insert(all_shifts, {sstart, send, sdest, cur, new_hyp, new_score, new_path})

    edits = edits + SHIFT_COST
    med_score = new_score
    med_path = new_path
    cur = new_hyp
  end

  return med_score + edits, med_path, cur, all_shifts
end

local function get_score_breakdown(path, shifts)
  -- Calculate the score breakdown by type
  -- INS DEL SUB SHIFT WORDS_SHIFTED
  local spieces = torch.Tensor{0, 0, 0, #shifts, 0}
  for _, e in ipairs(path) do
    if      e == "I" then
      spieces[1] = spieces[1] + 1
    elseif  e == "D" then
      spieces[2] = spieces[2] + 1
    elseif  e == "S" then
      spieces[3] = spieces[3] + 1
    end
  end
  for _, s in ipairs(shifts) do
    spieces[5] = spieces[5] + (s[2] - s[1]) + 1
  end
  return spieces
end

local function score_sent(id, HYP, REFS)
  -- try all references, and find the one with the lowest score
  -- return the score, path, shifts, etc
  local tmparr = {}
  local best_score = -1
  local best_ref = ""
  local best_path = ""
  local best_allshift = tmparr
  local best_hyp

  local rlen = 0

  for _, ref in ipairs(REFS) do
    rlen = rlen + #ref[id]
    local s, p, newhyp, allshifts = calc_shifts(HYP[id], ref[id])
    if best_score < 0 or s < best_score then
      best_score = s
      best_path = p
      best_ref = ref
      best_hyp = newhyp
      best_allshift = allshifts
    end
  end

  rlen = rlen / #REFS

  return best_score/rlen, best_path, best_ref, best_hyp, best_allshift, rlen
end

local function calculate_ter(cand, refs)
  local score = 0
  local num_tok = 0
  local score_breakdown = torch.Tensor(5):zero()
  for k,_ in ipairs(cand) do
    local best_score, best_path, _, _, best_allshift, rlen =
              score_sent(k, cand, refs)

    -- rlen is the average length of the reference (if there are multiple references)
    num_tok = num_tok + rlen

    -- document level score are based on the total number of tokens
    score = score + best_score * rlen
    local gsb = torch.Tensor(get_score_breakdown(best_path, best_allshift))
    score_breakdown:add(gsb * rlen)
  end
  score_breakdown:div(num_tok)
  local score_detail =
      string.format("TER = %.2f (Ins %.1f, Del %.1f, Sub %.1f, Shft %.1f, WdSh %.1f)",
          score*100/num_tok, score_breakdown[1], score_breakdown[2], score_breakdown[3],
          score_breakdown[4], score_breakdown[5]
      )
  return score/num_tok, score_detail
end

return calculate_ter
