local constants = require 'lib.utils.constants'

-- Convert a flat index to a row-column tuple.
local function flat_to_rc(v, flat_index)
  local row = math.floor((flat_index - 1) / v:size(2)) + 1
  return row, (flat_index - 1) % v:size(2) + 1
end


local Beam = torch.class('Beam')

function Beam:__init(size, max_seq_len)
  self.size = size
  self.done = false
  self.found_eos = false

  self.scores = torch.FloatTensor(size):zero()
  self.prev_ks = { torch.LongTensor(size):fill(1) }
  self.next_ys = { torch.LongTensor(size):fill(constants.PAD) }
  self.attn = {}
  self.next_ys[1][1] = constants.BOS
end

function Beam:get_current_state()
  return self.next_ys[#self.next_ys]
end

function Beam:get_current_origin()
  return self.prev_ks[#self.prev_ks]
end

function Beam:advance(out, attn_out)
  local flat_out

  if #self.prev_ks > 1 then
    for k = 1, self.size do
      out[k]:add(self.scores[k])
    end

    flat_out = out:view(-1)
  else
    flat_out = out[1]:view(-1)
  end

  local prev_k = torch.LongTensor(self.size)
  local next_y = torch.LongTensor(self.size)
  local attn = {}

  local best_scores, best_scores_id = flat_out:topk(self.size, 1, true, true)

  for k = 1, self.size do
    self.scores[k] = best_scores[k]

    local from_beam, best_score_id = flat_to_rc(out, best_scores_id[k])

    prev_k[k] = from_beam
    next_y[k] = best_score_id
    table.insert(attn, attn_out[from_beam]:clone())

    if next_y[k] == constants.EOS then
      self.found_eos = true
      if k == 1 then
        self.done = true
      end
    end
  end

  table.insert(self.prev_ks, prev_k)
  table.insert(self.next_ys, next_y)
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

function Beam:get_hyp(k)
  local hyp = {}
  local attn = {}

  for _ = 1, #self.prev_ks - 1 do
    table.insert(hyp, {})
    table.insert(attn, {})
  end

  for j = #self.prev_ks, 2, -1 do
    hyp[j - 1] = self.next_ys[j][k]
    attn[j - 1] = self.attn[j - 1][k]
    k = self.prev_ks[j][k]
  end

  return hyp, attn
end

return Beam
