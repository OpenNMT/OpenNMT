local Beam = torch.class('Beam')

function Beam:__init(tokens, states)
  self._tokens = tokens
  self._states = states
end

function Beam:tokens()
  return self._tokens
end

function Beam:flatTokens()
  return self._tokens:view(-1)
end
function Beam:states()
  return self._states
end

function Beam:scores()
  return self._scores
end

function Beam:backPointers()
  return self._backPointers
end

-- binary vector
function Beam:batchesFinished()
  return self._batchesFinished
end

-- dict
function Beam:addCompletedHypotheses()
  table.insert(self.completed, {}) -- batch, pointer, t, score, etc.
end
