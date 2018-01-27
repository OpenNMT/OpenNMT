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
  advancer.dicts = {}
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

  -- Test SubDict mapping
  advancer.dicts.subdict = onmt.utils.SubDict
  advancer.dicts.subdict.vocabs = {1,2,3}
  advancer.dicts.subdict.targetVocInvMap = torch.LongTensor{2,1,4,3}
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results[1][1].tokens, {1, 1, 1})
  advancer.dicts.subdict = nil

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
  local filter = function(beam, consideredToken, _, consideredBackPointer)
    local prevTokens = beam:getTokens()

    local consTokens = consideredToken:view(-1)
    local consBackPtr = consideredBackPointer:view(-1)

    local batchByBeamSize = consTokens:size(1)
    -- Disallow {3, 4}
    local prune = torch.ByteTensor(batchByBeamSize):zero()
    for b = 1, batchByBeamSize do
      if #prevTokens == 2 then
        if prevTokens[#prevTokens][consBackPtr[b]] == 3 and consTokens[b] == 4 then
          prune[b] = 1
        end
      end
    end
    return prune
  end
  Advancer.filter = function(_, beam, consideredToken, _, consideredBackPointer) return filter(beam, consideredToken, _, consideredBackPointer) end
  advancer = Advancer.new()
  advancer.dicts = {}
  nBest = 1
  beamSize = 3
  beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  results = beamSearcher:search(beamSize, nBest)
  tester:eq(results, { {{tokens = {2, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {1, 3, 4}, states = {}, score = math.log(.6*.4*.9)}},
                       {{tokens = {4}, states = {}, score = math.log(.9)}} }, 1e-6)

  -- Test beam search saver.
  beamSearcher = onmt.translate.BeamSearcher.new(advancer, true)
  local _, histories = beamSearcher:search(beamSize, nBest)
  tester:eq(#histories[1].predictedIds, 2)
end

return beamSearchTest
