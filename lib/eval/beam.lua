local constants = require 'lib.constants'

--[[Helper function convert a `flat_index` to a row-column tuple

Parameters:

  * `v` - matrix.
  * `flat_index` - index

Returns: row/column.
--]]
local function flat_to_rc(v, flat_index)
  local row = math.floor((flat_index - 1) / v:size(2)) + 1
  return row, (flat_index - 1) % v:size(2) + 1
end

--[[ Class for managing the beam search process. ]]
local Beam = torch.class('Beam')

--[[Constructor

Parameters:

  * `size` : The beam `K`.
--]]
function Beam:__init(size, num_features)

  self.size = size
  self.num_features = num_features
  self.done = false

  -- The score for each translation on the beam.
  self.scores = torch.FloatTensor(size):zero()

  -- The backpointers at each time-step.
  self.prev_ks = { torch.LongTensor(size):fill(1) }

  -- The outputs at each time-step.
  self.next_ys = { torch.LongTensor(size):fill(constants.PAD) }
  self.next_ys[1][1] = constants.BOS

  -- The features output at each time-step
  self.next_features = { {} }
  for j = 1, num_features do
    self.next_features[1][j] = torch.LongTensor(size):fill(constants.PAD)

    -- EOS is used as a placeholder to shift the features target sequence.
    self.next_features[1][j][1] = constants.EOS
  end

  -- The attentions (matrix) for each time.
  self.attn = {}
end

--[[ Get the outputs for the current timestep.]]
function Beam:get_current_state()
  return self.next_ys[#self.next_ys], self.next_features[#self.next_features]
end

--[[ Get the backpointers for the current timestep.]]
function Beam:get_current_origin()
  return self.prev_ks[#self.prev_ks]
end

--[[ Given prob over words for every last beam `word_lk` and attention
 `attn_out`. Compute and update the beam search.

Parameters:

  * `word_lk`- probs at the last step
  * `attn_word_lk`- attention at the last step

Returns: true if beam search is complete.
--]]
function Beam:advance(word_lk, feats_lk, attn_out)

  -- The flattened scores.
  local flat_word_lk

  if #self.prev_ks > 1 then
    -- Sum the previous scores.
    for k = 1, self.size do
      word_lk[k]:add(self.scores[k])
    end
    flat_word_lk = word_lk:view(-1)
  else
    flat_word_lk = word_lk[1]:view(-1)
  end


  -- Find the top-k elements in flat_word_lk and backpointers.
  local prev_k = torch.LongTensor(self.size)
  local next_y = torch.LongTensor(self.size)
  local next_feat = {}
  local attn = {}

  for j = 1, #feats_lk do
    next_feat[j] = torch.LongTensor(self.size)
  end

  local best_scores, best_scores_id = flat_word_lk:topk(self.size, 1, true, true)

  for k = 1, self.size do
    self.scores[k] = best_scores[k]

    local from_beam, best_score_id = flat_to_rc(word_lk, best_scores_id[k])

    prev_k[k] = from_beam
    next_y[k] = best_score_id
    table.insert(attn, attn_out[from_beam]:clone())

    -- For features, just store predictions for each beam.
    for j = 1, #feats_lk do
      local _, best = feats_lk[j]:max(2)
      next_feat[j]:copy(best)
    end
  end

  -- End condition is when top-of-beam is EOS.
  if next_y[1] == constants.EOS then
    self.done = true
  end

  table.insert(self.prev_ks, prev_k)
  table.insert(self.next_ys, next_y)
  table.insert(self.next_features, next_feat)
  table.insert(self.attn, attn)

  return self.done
end

function Beam:sort_best()
  return torch.sort(self.scores, 1, true)
end

function Beam:get_best()
  local scores, ids = self:sort_best()
  return scores[1], ids[1]
end

--[[ Walk back to construct the full hypothesis `k`.

Parameters:

  * `k` - the position in the beam to construct.

Returns:

  1. The hypothesis
  2. The attention at each time step.
--]]
function Beam:get_hyp(k)
  local hyp = {}
  local feats = {}
  local attn = {}

  for _ = 1, #self.prev_ks - 1 do
    table.insert(hyp, {})
    table.insert(attn, {})

    if self.num_features > 0 then
      table.insert(feats, {})
    end
  end

  for j = #self.prev_ks, 2, -1 do
    hyp[j - 1] = self.next_ys[j][k]
    for i = 1, self.num_features do
      feats[j - 1][i] = self.next_features[j][i][k]
    end
    attn[j - 1] = self.attn[j - 1][k]
    k = self.prev_ks[j][k]
  end

  return hyp, feats, attn
end

return Beam
