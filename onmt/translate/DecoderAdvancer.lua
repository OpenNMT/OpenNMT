--[[ DecoderAdvancer is an implementation of the interface Advancer for
  specifyinghow to advance one step in decoder.
--]]
local DecoderAdvancer = torch.class('DecoderAdvancer', 'Advancer')

--[[ Constructor.

Parameters:

  * `decoders` - a table of decoders or nil to rely on threads' context.
  * `batch` - an `onmt.data.Batch` object.
  * `contexts` - a table of encoder outputs (batch x T x rnnSize).
  * `max_sent_length` - optional, maximum output sentence length.
  * `max_num_unks` - optional, maximum number of UNKs.
  * `decStates` - optional, a tablea of initial decoder states.
  * `dicts` - optional, dictionary for additional features.

--]]
function DecoderAdvancer:__init(
    decoders,
    batch,
    contexts,
    max_sent_length,
    max_num_unks,
    decStates,
    dicts,
    length_norm,
    coverage_norm,
    eos_norm)
  self.dicts = dicts
  self.batch = batch
  self.max_sent_length = max_sent_length or math.huge
  self.max_num_unks = max_num_unks or math.huge
  self.length_norm = length_norm or 0.0
  self.coverage_norm = coverage_norm or 0.0
  self.eos_norm = eos_norm or 0.0

  self.decoders = decoders or {}
  self.contexts = type(contexts) == 'table' and contexts or { contexts }

  if decStates then
    self.decStates = type(decStates) == 'table' and decStates or { decStates }
  else
    -- Generate states for each decoder.
    self.decStates = {}

    onmt.utils.ThreadPool.dispatch(
      function(i)
        local decoder = decoders[i] or _G.model.models.decoder
        local states = onmt.utils.Tensor.initTensorTable(
          decoder.args.numEffectiveLayers,
          onmt.utils.Cuda.convert(torch.Tensor()),
          { self.batch.size, decoder.args.rnnSize })
        return i, states
      end,
      function(i, states)
        self.decStates[i] = states
      end
    )
  end

  self.attnBuffer = onmt.utils.Cuda.convert(torch.Tensor())
  self.logProbsBuffer = onmt.utils.Cuda.convert(torch.Tensor())
end

--[[Returns an initial beam.

Returns:

  * `beam` - an `onmt.translate.Beam` object.

--]]
function DecoderAdvancer:initBeam()
  local tokens = onmt.utils.Cuda.convert(torch.IntTensor(self.batch.size)):fill(onmt.Constants.BOS)
  local features = {}
  if self.dicts then
    for j = 1, #self.dicts.tgt.features do
      features[j] = onmt.utils.Cuda.convert(torch.IntTensor(self.batch.size):fill(onmt.Constants.EOS))
    end
  end
  local sourceSizes = onmt.utils.Cuda.convert(self.batch.sourceSize)
  local attnProba = torch.FloatTensor(self.batch.size, self.contexts[1]:size(2))
    :fill(0.0001)
    :typeAs(self.contexts[1])
  -- Assign maximum attention proba on padding for it to not interfer during coverage normalization.
  for i = 1, self.batch.size do
    local sourceSize = sourceSizes[i]
    if self.batch.sourceLength ~= self.contexts[1]:size(2) then
      sourceSize = math.ceil(sourceSize / (self.batch.sourceLength / self.contexts[1]:size(2)))
    end
    local padSize = self.contexts[1]:size(2) - sourceSize
    if padSize ~= 0 then
      attnProba[{i, {1, padSize}}] = 1.0
    end
  end

  -- Define state to be { decoder states, decoder output, context,
  -- attentions, features, sourceSizes, step, cumulated attention probablities }.
  local state = { self.decStates, {}, self.contexts, nil, features, sourceSizes, 1, attnProba }
  local params = {}
  params.length_norm = self.length_norm
  params.coverage_norm = self.coverage_norm
  params.eos_norm = self.eos_norm
  return onmt.translate.Beam.new(tokens, state, params)
end

--[[Updates beam states given new tokens.

Parameters:

  * `beam` - beam with updated token list.

]]
function DecoderAdvancer:update(beam)
  local state = beam:getState()
  local decStates, decOut, contexts, _, features, sourceSizes, t, cumAttnProba
    = table.unpack(state, 1, 8)
  local tokens = beam:getTokens()
  local token = tokens[#tokens]
  local inputs
  if #features == 0 then
    inputs = token
  elseif #features == 1 then
    inputs = { token, features[1] }
  else
    inputs = { token }
    table.insert(inputs, features)
  end

  local attentions = {}

  local sourceLength = self.batch.sourceLength
  local decoders = self.decoders

  -- Run decoders in parallel.
  onmt.utils.ThreadPool.dispatch(
    function(i)
      local decoder = decoders[i] or _G.model.models.decoder

      local out, states = decoder:forwardOne(onmt.utils.Tensor.recursiveClone(inputs),
                                             decStates[i],
                                             contexts[i],
                                             decOut[i],
                                             nil,
                                             sourceSizes,
                                             sourceLength)

      local attention = decoder:getAttention()

      return i, out, states, attention
    end,
    function(i, out, states, attention)
      decOut[i] = out
      decStates[i] = states
      attentions[i] = attention
    end
  )

  if attentions[1] then
    self.attnBuffer:resizeAs(attentions[1])

    -- Accumulate attention as an average.
    for i = 2, #attentions do
      self.attnBuffer:copy(attentions[i])
      attentions[1]:mul(i - 1):add(self.attnBuffer):div(i)
    end

    cumAttnProba:add(attentions[1])
  end

  t = t + 1

  local nextState = {decStates, decOut, contexts, attentions[1], nil, sourceSizes, t, cumAttnProba}
  beam:setState(nextState)
end

--[[Expand function. Expands beam by all possible tokens and returns the
  scores.

Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns:

  * `scores` - a 2D tensor of size `(batchSize * beamSize, numTokens)`.

]]
function DecoderAdvancer:expand(beam)
  local state = beam:getState()
  local decOut = state[2]

  local decoders = self.decoders
  local logProbs = {}

  onmt.utils.ThreadPool.dispatch(
    function(i)
      local decoder = decoders[i] or _G.model.models.decoder
      return i, decoder.generator:forward(decOut[i])
    end,
    function(i, logProb)
      logProbs[i] = logProb
    end
  )

  for i = 2, #logProbs do
    for j = 1, #logProbs[i] do
      self.logProbsBuffer:resizeAs(logProbs[i][j])
      self.logProbsBuffer:copy(logProbs[i][j])

      logProbs[1][j]
        :exp()
        :mul(i - 1)
        :add(self.logProbsBuffer:exp())
        :div(i)
        :log()
    end
  end

  local features = {}
  for j = 2, #logProbs[1] do
    local _, best = logProbs[1][j]:max(2)
    features[j - 1] = best:view(-1)
  end
  state[5] = features
  local scores = logProbs[1][1]
  return scores
end

--[[Checks which hypotheses in the beam are already finished. A hypothesis is
  complete if i) an onmt.Constants.EOS is encountered, or ii) the length of the
  sequence is greater than or equal to `max_sent_length`.

Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns: a binary flat tensor of size `(batchSize * beamSize)`, indicating
  which hypotheses are finished.

]]
function DecoderAdvancer:isComplete(beam)
  local tokens = beam:getTokens()
  local seqLength = #tokens - 1
  local complete = tokens[#tokens]:eq(onmt.Constants.EOS)
  if seqLength > self.max_sent_length then
    complete:fill(1)
  end
  return complete
end

--[[Checks which hypotheses in the beam shall be pruned. We disallow empty
 predictions, as well as predictions with more UNKs than `max_num_unks`.

Parameters:

  * `beam` - an `onmt.translate.Beam` object.

Returns: a binary flat tensor of size `(batchSize * beamSize)`, indicating
  which beams shall be pruned.

]]
function DecoderAdvancer:filter(beam)
  local tokens = beam:getTokens()
  local numUnks = onmt.utils.Cuda.convert(torch.zeros(tokens[1]:size(1)))
  for t = 1, #tokens do
    local token = tokens[t]
    numUnks:add(onmt.utils.Cuda.convert(token:eq(onmt.Constants.UNK):double()))
  end

  -- Disallow too many UNKs
  local pruned = numUnks:gt(self.max_num_unks)

  -- Disallow empty hypotheses
  if #tokens == 2 then
    pruned:add(tokens[2]:eq(onmt.Constants.EOS))
  end
  return pruned:ge(1)
end

return DecoderAdvancer
