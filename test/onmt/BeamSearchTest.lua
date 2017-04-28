require('onmt.init')

local tester = ...

local beamSearchTest = torch.TestSuite()

function beamSearchTest.beamSearch()
  local transitionScores = { {-math.huge, math.log(.6), math.log(.4), -math.huge},
                   {math.log(.6), -math.huge, math.log(.4), -math.huge},
                   {-math.huge, -math.huge, math.log(.1), math.log(.9)},
                   {-math.huge, -math.huge, -math.huge, -math.huge}
               }
  transitionScores = torch.Tensor(transitionScores)

  local Advancer = onmt.translate.Advancer

  local initBeam = function()
    return onmt.translate.Beam.new(torch.LongTensor({1, 2, 3}), {})
  end
  local update = function()
  end
  local expand = function(beam)
    local tokens = beam:getTokens()
    local token = tokens[#tokens]
    local scores = transitionScores:index(1, token)
    return scores
  end
  local isComplete = function(beam)
    local tokens = beam:getTokens()
    local completed = tokens[#tokens]:eq(4)
    if #tokens - 1 > 2 then
      completed:fill(1)
    end
    return completed
  end

  Advancer.initBeam = function() return initBeam() end
  Advancer.update = function(_, beam) update(beam) end
  Advancer.expand = function(_, beam) return expand(beam) end
  Advancer.isComplete = function(_, beam) return isComplete(beam) end
  local beamSize, nBest, advancer, beamSearcher, results
  advancer = Advancer.new()
  -- Test different beam sizes
  nBest = 1
  -- Beam size 2
  beamSize = 2
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)}},
                       {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)}} }, 1e-6)
  -- Beam size 1
  beamSize = 1
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {2, 1, 2}, states = {}, score = math.log(.6*.6*.6)}},
                       {{tokens = {1, 2, 1}, states = {}, score = math.log(.6*.6*.6)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)}} }, 1e-6)

  -- Test nBest = 2
  nBest = 2
  beamSize = 3
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)},
                        {tokens = {2, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {3, 4}, states = {}, score = math.log(.4*.9)},
                        {tokens = {1, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)},
                       {tokens = {3, 4}, states = {}, score = math.log(.1*.9)}} }, 1e-6)

  -- Test filter
  local filter = function(beam)
    local tokens = beam:getTokens()
    local batchSize = tokens[1]:size(1)
    -- Disallow {3, 4}
    local prune = torch.ByteTensor(batchSize):zero()
    for b = 1, batchSize do
      if #tokens >= 3 then
        if tokens[2][b] == 3 and tokens[3][b] == 4 then
          prune[b] = 1
        end
      end
    end
    return prune
  end
  Advancer.filter = function(_, beam) return filter(beam) end
  advancer = Advancer.new()
  nBest = 1
  beamSize = 3
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {2, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {1, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)}} }, 1e-6)
end

return beamSearchTest
