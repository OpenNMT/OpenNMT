--[[ DecoderAdvancer is an implementation of the interface Advancer for
  specifying how to advance one step in decoder.
--]]
local DecoderAdvancer = torch.class('DecoderAdvancer', 'Advancer')

--[[ Constructor.

Parameters:

  * `decoder` - an `onmt.Decoder` object.
  * `batch` - an `onmt.data.Batch` object.
  * `context` - encoder output (batch x n x rnnSize).
  * `max_sent_length` - optional, maximum output sentence length.
  * `max_num_unks` - optional, maximum number of UNKs.
  * `decStates` - optional, initial decoder states.
  * `lmModel` - optional, the language model object.
  * `lmStates`, `lmContext` - option initial language model states and context - initialized with BOS
  * `dicts` - optional, dictionary for additional features.

--]]
function DecoderAdvancer:__init(decoder, batch, context, max_sent_length, max_num_unks, decStates,
                                lmModel, lmStates, lmContext, lm_weight,
                                dicts, length_norm, coverage_norm, eos_norm)
  self.decoder = decoder
  self.batch = batch
  self.context = context
  self.max_sent_length = max_sent_length or math.huge
  self.max_num_unks = max_num_unks or math.huge
  self.length_norm = length_norm or 0.0
  self.coverage_norm = coverage_norm or 0.0
  self.eos_norm = eos_norm or 0.0
  self.decStates = decStates or onmt.utils.Tensor.initTensorTable(
    decoder.args.numStates,
    onmt.utils.Cuda.convert(torch.Tensor()),
    { self.batch.size, decoder.args.rnnSize })
  self.lmModel = lmModel
  self.lmStates = lmStates
  self.lmContext = lmContext
  self.lm_weight = lm_weight
  self.dicts = dicts
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
  local attnProba = torch.FloatTensor(self.batch.size, self.context:size(2))
    :fill(0.0001)
    :typeAs(self.context)
  -- Assign maximum attention proba on padding for it to not interfer during coverage normalization.
  for i = 1, self.batch.size do
    local sourceSize = sourceSizes[i]
    if self.batch.sourceLength ~= self.context:size(2) then
      sourceSize = math.ceil(sourceSize / (self.batch.sourceLength / self.context:size(2)))
    end
    local padSize = self.context:size(2) - sourceSize
    if padSize ~= 0 then
      attnProba[{i, {1, padSize}}] = 1.0
    end
  end

  -- Define state to be { decoder states, decoder output, context,
  -- attentions, features, sourceSizes, step, , lmStates, lmContext, lexical constraints, lexical constraintSizes }.
  local state = { self.decStates,             -- idx 1  : decoder states
                  nil,                        -- idx 2  : decoder output
                  self.context,               -- idx 3  : context
                  nil,                        -- idx 4  : attentions
                  features,                   -- idx 5  : features
                  sourceSizes,                -- idx 6  : sourceSizes
                  1,                          -- idx 7  : step
                  attnProba,                  -- idx 8  : cumulated attention probablities
                  self.lmStates,              -- idx 9  : lmStates
                  self.lmContext,             -- idx 10 : lmContext
                  self.batch.constraints,     -- idx 11 : lexical constraints remaining to apply to this node
                  self.batch.constraintSizes  -- idx 12 : lexical constraints sizes
  }

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
  local decStates, decOut, context, _, features, sourceSizes, t, cumAttnProba, lmStates, lmContext, constraints, constraintSizes
    = table.unpack(state, 1, 12)

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

  -- compute next decoder step
  decOut, decStates = self.decoder:forwardOne(inputs,
                                              decStates,
                                              context,
                                              decOut,
                                              nil,
                                              sourceSizes,
                                              self.batch.sourceLength)

  -- if defined, compute next language model step
  if self.lmModel then
    lmStates, lmContext = self.lmModel.encoder:forwardOne(inputs, lmStates, true)
  end

  t = t + 1

  local attention = self.decoder:getAttention()
  if attention then
    cumAttnProba = cumAttnProba:add(attention)
  end

  local nextState = {decStates, decOut, context, attention, nil, sourceSizes, t, cumAttnProba, lmStates, lmContext, constraints, constraintSizes}
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
  local out = self.decoder.generator:forward(decOut)
  local features = {}
  for j = 2, #out do
    local _, best = out[j]:max(2)
    features[j - 1] = best:view(-1)
  end
  state[5] = features
  local scores = out[1]

  if self.lmModel then
    local lmOut = self.lmModel.generator:forward(state[10])
    scores = scores + lmOut[1] * self.lm_weight
  end

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

--[[Checks which hypotheses shall be pruned. We disallow empty
 predictions, as well as predictions with more UNKs than `max_num_unks`.

Parameters:

  * `beam` current beam
  * `consideredToken` hypotheses tokens
  * `consideredScores` hypotheses scores
  * `consideredBackPointer` back pointer

Returns: a binary flat tensor indicating which beams shall be pruned.

]]
function DecoderAdvancer:filter(beam, consideredToken, _, consideredBackPointer)
  local tokens = beam:getTokens()
  local numUnks = onmt.utils.Cuda.convert(torch.zeros(tokens[1]:size(1)))
  for t = 1, #tokens do
    local token = tokens[t]
    numUnks:add(onmt.utils.Cuda.convert(token:eq(onmt.Constants.UNK):double()))
  end

  local toks = consideredToken:view(-1)
  local backPtr = consideredBackPointer:view(-1)

  local pruned = onmt.utils.Cuda.convert(torch.zeros(toks:size(1)))

  for i = 1, toks:size(1) do
    local tok = toks[i]
    if (tok == onmt.Constants.UNK and numUnks[backPtr[i]] >= self.max_num_unks) or
       (#tokens == 1 and tok == onmt.Constants.EOS) then
      pruned[i] = 1
    end
  end
  return pruned:ge(1)
end

return DecoderAdvancer
