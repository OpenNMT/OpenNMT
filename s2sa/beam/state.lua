require 'torch'
local table_utils = require 's2sa.table_utils'

local BeamState = torch.class("BeamState")

function BeamState.initial(start)
  return {start}
end

function BeamState.advance(state, token)
  local new_state = table_utils.copy(state)
  table.insert(new_state, token)
  return new_state
end

function BeamState.disallow(out)
  local bad = {1, 3} -- 1 is PAD, 3 is BOS
  for j = 1, #bad do
    out[bad[j]] = -1e9
  end
end

function BeamState.same(state1, state2)
  for i = 2, #state1 do
    if state1[i] ~= state2[i] then
      return false
    end
  end
  return true
end

function BeamState.next(state)
  return state[#state]
end

function BeamState.heuristic()
  return 0
end

function BeamState.print(state)
  for i = 1, #state do
    io.write(state[i] .. " ")
  end
  print()
end

return BeamState
