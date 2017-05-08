--[[ DecoderAdvancer is an implementation of the interface Advancer for
  specifyinghow to advance one step in decoder.
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
  * `dicts` - optional, dictionary for additional features.
  * `updateSeqLengthFunc` - optional, sequence length adaptation function after encoder

--]]
function DecoderAdvancer:__init(decoder, batch, context, max_sent_length, max_num_unks, decStates, dicts, length_norm, coverage_norm, eos_norm, updateSeqLengthFunc)
  self.decoder = decoder
  self.batch = batch
  self.context = context
  self.max_sent_length = max_sent_length or math.huge
  self.max_num_unks = max_num_unks or math.huge
  self.length_norm = length_norm or 0.0
  self.coverage_norm = coverage_norm or 0.0
  self.eos_norm = eos_norm or 0.0
  self.decStates = decStates or onmt.utils.Tensor.initTensorTable(
    decoder.args.numEffectiveLayers,
    onmt.utils.Cuda.convert(torch.Tensor()),
    { self.batch.size, decoder.args.rnnSize })
  self.dicts = dicts
  self.updateSeqLengthFunc = updateSeqLengthFunc
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
      features[j] = torch.IntTensor(self.batch.size):fill(onmt.Constants.EOS)
    end
  end
  local sourceSizes = onmt.utils.Cuda.convert(self.batch.sourceSize)
  local attnProba = torch.FloatTensor(self.batch.size, self.context:size(2))
    :fill(0.0001)
    :typeAs(self.context)
  -- Mask padding
  for i = 1,self.batch.size do
    local pad_size = self.context:size(2) - sourceSizes[i]
    if (pad_size ~= 0) then
      attnProba[{ i, {1,pad_size} }] = 1.0
    end
  end

  -- Define state to be { decoder states, decoder output, context,
  -- attentions, features, sourceSizes, step, cumulated attention probablities }.
  local state = { self.decStates, nil, self.context, nil, features, sourceSizes, 1, attnProba }
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
  local decStates, decOut, context, _, features, sourceSizes, t, cumAttnProba
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

  local contextSizes, contextLength = sourceSizes, self.batch.sourceLength
  if self.updateSeqLengthFunc then
    contextSizes, contextLength = self.updateSeqLengthFunc(contextSizes, contextLength)
  end

  self.decoder:maskPadding(contextSizes, contextLength)
  decOut, decStates = self.decoder:forwardOne(inputs, decStates, context, decOut)
  t = t + 1

  local softmaxOut

  if self.decoder.softmaxAttn then
    softmaxOut = self.decoder.softmaxAttn.output
    cumAttnProba = cumAttnProba:add(softmaxOut)
  end

  local nextState = {decStates, decOut, context, softmaxOut, nil, sourceSizes, t, cumAttnProba}
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
