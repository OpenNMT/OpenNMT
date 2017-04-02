--[[ DecoderAdvancer is an implementation of the interface Advancer for
  specifyinghow to advance one step in decoder.
--]]
local DecoderAdvancer = torch.class('EnsembleDecoderAdvancer', 'Advancer')

--[[ Constructor.

Parameters:

  * `decoders` - a set of `onmt.Decoder` object.
  * `batch` - an `onmt.data.Batch` object.
  * `contexts` - encoder output nmodels x (batch x n x rnnSize).
  * `max_sent_length` - optional, maximum output sentence length.
  * `max_num_unks` - optional, maximum number of UNKs.
  * `decStates` - optional, a set of initial decoder states.
  * `dicts` - optional, dictionary for additional features.
  * `word_pen` - optional, penalty value to favor longer sentences

--]]
function DecoderAdvancer:__init(decoders, batch, contexts, max_sent_length, max_num_unks, decStates, dicts, word_pen, ensembleOps)
  self.decoders = decoders
  self.batch = batch
  self.contexts = contexts
  self.max_sent_length = max_sent_length or math.huge
  self.max_num_unks = max_num_unks or math.huge
  self.nModels = #self.decoders -- number of models
  self.ensembleOps = ensembleOps or 'sum' -- sum or logsum
  
  self.decStates = {}
  
  -- initialize the decStates, a bit tricky
  for i = 1, self.nModels do
	self.decStates[i] = decStates[i] or onmt.utils.Tensor.initTensorTable(
    decoders[1].args.numEffectiveLayers,
    onmt.utils.Cuda.convert(torch.Tensor()),
    { self.batch.size, decoders[1].args.rnnSize })
  end
  --~ self.decStates = decStates or onmt.utils.Tensor.initTensorTable(
    --~ decoder.args.numEffectiveLayers,
    --~ onmt.utils.Cuda.convert(torch.Tensor()),
    --~ { self.batch.size, decoder.args.rnnSize })
  self.dicts = dicts
  self.word_pen = word_pen or 0
  self.logSoftMax = nn.LogSoftMax()
  onmt.utils.Cuda.convert(self.logSoftMax)
  
end

function DecoderAdvancer:ensembleScore(scores)
	
	
	local score = scores[1]	
	local nOutputs = #score
	
	
	
	if self.ensembleOps == 'sum' then -- get the average of the probability
		for n = 1, nOutputs do
			score[n] = torch.exp(score[n]) -- so we have to exp to get the prob
			for i = 2, self.nModels do
				score[n]:add(torch.exp(scores[i][n]))
			end
			score[n]:div(self.nModels) -- take the average
			score[n] = torch.log(score[n])
		end
	else -- logsum operation. 
		for n = 1, nOutputs do
			for i = 2, self.nModels do
				score[n]:add(scores[i][n])
			end
			score[n]:div(self.nModels)
			score[n] = self.logSoftMax:forward(score[n])
		end 
	end
	
	return score

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

  -- Define state to be { {decoder states}, {decoder output}, {decoder coverage}, {context},
  -- {attentions}, features, sourceSizes, step }.
  local state = { self.decStates, nil, nil, self.contexts, nil, features, sourceSizes, 1 }
  return onmt.translate.Beam.new(tokens, state)
end

--[[Updates beam states given new tokens.

Parameters:

  * `beam` - beam with updated token list.

]]
function DecoderAdvancer:update(beam)
  local state = beam:getState()
  local decStates, decOuts, decCovs, contexts, _, features, sourceSizes, t
    = table.unpack(state, 1, 8)
  --~ print(features)
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
  
  local newOuts = {}
  local newCovs = {}
  local newStates = {}
  local attnOuts = {}
  for i = 1, self.nModels do
	self.decoders[i]:maskPadding(sourceSizes, self.batch.sourceLength)
	
	local decOut = decOuts and decOuts[i] or nil
	local decCov = decCovs and decCovs[i] or nil
	local context = contexts[i]
	local decState = decStates and decStates[i] or nil
	
	-- input is always the same for the decoders
	decOut, decCov, decState = self.decoders[i]:forwardOne(inputs, decState, context, decOut, decCov)
	
	newOuts[i] = decOut
	newCovs[i] = decCov -- could be nil
	newStates[i] = decState
	
	local softmaxOut = self.decoders[i].softmaxAttn.output
	attnOuts[i] = softmaxOut
  end
 
  
  t = t + 1
  local nextState = {newStates, newOuts, newCovs, contexts, attnOuts, nil, sourceSizes, t}
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
  local decOuts = state[2]
  
  -- Where the output distribution is generated by the decoder
  --~ local out = self.decoder.generator:forward(decOut)
  local logSoftMaxes = {}
  for i = 1, self.nModels do
	logSoftMaxes[i] = self.decoders[i].generator:forward(decOuts[i])
  end
  local out = self:ensembleScore(logSoftMaxes) -- average the score 
  
  local features = {}
  
  -- Adding word penalty to get sentences with better length
  -- We can also consider normalizing Prob with length but this is easy
  out[1] = out[1] + self.word_pen
  
  for j = 2, #out do
	local _, best = out[j]:max(2)
    features[j - 1] = best:view(-1)
  end
  state[6] = features
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
