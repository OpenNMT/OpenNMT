local Translator = torch.class('EnsembleTranslator')
local pl = require('pl.import_into')()

local function clearStateModel(model)
  for _, submodule in pairs(model.modules) do
    if torch.type(submodule) == 'table' and submodule.modules then
      clearStateModel(submodule)
    else
      submodule:clearState()
      submodule:apply(function (m)
        nn.utils.clear(m, 'gradWeight', 'gradBias')
      end)
    end
  end
end


local options = {
  {'-model', '', [[Path to model .t7 files, separated by |]], {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-beam_size', 5, [[Beam size]]},
  {'-batch_size', 30, [[Batch size]]},
  {'-word_pen', 0, [[Word Penalty during decoding]]},
  {'-max_sent_length', 250, [[Maximum output sentence length.]]},
  {'-replace_unk', false, [[Replace the generated UNK tokens with the source token that
                          had the highest attention weight. If phrase_table is provided,
                          it will lookup the identified source token and give the corresponding
                          target token. If it is not provided (or the identified source token
                          does not exist in the table) then it will copy the source token]]},
  {'-phrase_table', '', [[Path to source-target dictionary to replace UNK
                        tokens. See README.md for the format this file should be in]]},
  {'-n_best', 1, [[If > 1, it will also output an n_best list of decoded sentences]]},
  {'-save_mem', 1, [[If = 1, it will clear the model states to reduce memory load. However the loading state could be a bit slower.]]},
  {'-max_num_unks', math.huge, [[All sequences with more unks than this will be ignored
                               during beam search]]},
  {'-pre_filter_factor', 1, [[Optional, set this only if filter is being used. Before
                            applying filters, hypotheses with top `beamSize * preFilterFactor`
                            scores will be considered. If the returned hypotheses voilate filters,
                            then set this to a larger value to consider more.]]}
}

function Translator.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'EnsembleTranslator')
end


function Translator:__init(args)
  self.opt = args
  onmt.utils.Cuda.init(self.opt)
  
  
  local models = pl.utils.split(self.opt.model, '|')
  
  local nModels = #models
  
  self.models = {}

  --~ self.checkpoints = {}
 
  --~ for i = 1, nModels do
	--~ _G.logger:info('Loading \'' .. models[i] .. '\'...')
		--~ self.checkpoints[i] = torch.load(models[i])
  --~ end 
  
  --~ _G.logger:info('Checking Vocabularies ...')
  
  --~ local srcVocabSize = self.checkpoints[1].dicts.src.words:size()
  --~ local tgtVocabSize = self.checkpoints[1].dicts.tgt.words:size()
  --~ for i = 2, nModels do
		--~ assert(self.checkpoints[i].dicts.src.words:size() == srcVocabSize)
		--~ assert(self.checkpoints[i].dicts.tgt.words:size() == tgtVocabSize)
  --~ end
  
  --~ local dicts
	for i = 1, nModels do
		_G.logger:info('Loading \'' .. models[i] .. '\'...')
		
		local checkpoint = torch.load(models[i])
		
		
		-- checking vocabularies with the same size
		if i == 1 then
			self.dicts = checkpoint.dicts
		else
			local srcVocabSize = checkpoint.dicts.src.words:size()
			local tgtVocabSize = checkpoint.dicts.tgt.words:size()
			
			assert(self.dicts.src.words:size() == srcVocabSize)
			assert(self.dicts.tgt.words:size() == tgtVocabSize)
		end
		
		self.models[i] = {}
		self.models[i].encoder = onmt.Factory.loadEncoder(checkpoint.models.encoder)
		self.models[i].decoder = onmt.Factory.loadDecoder(checkpoint.models.decoder)
		
		clearStateModel(self.models[i].encoder)
		clearStateModel(self.models[i].decoder)
		-- save memory
		checkpoint = nil
		collectgarbage()
		
		self.models[i].encoder:evaluate()
		self.models[i].decoder:evaluate()
		onmt.utils.Cuda.convert(self.models[i].encoder)
		onmt.utils.Cuda.convert(self.models[i].decoder)
	end
  
  _G.logger:info('Done...')
  
  

  if self.opt.phrase_table:len() > 0 then
    self.phraseTable = onmt.translate.PhraseTable.new(self.opt.phrase_table)
  end
  
  self.nModels = nModels
  
  self.logSoftMax = nn.LogSoftMax()
  onmt.utils.Cuda.convert(self.logSoftMax)
  self.ensembleOps = 'sum'
end

function Translator:buildInput(tokens)
  local words, features = onmt.utils.Features.extract(tokens)

  local data = {}
  data.words = words

  if #features > 0 then
    data.features = features
  end

  return data
end

function Translator:buildOutput(data)
  return table.concat(onmt.utils.Features.annotate(data.words, data.features), ' ')
end

function Translator:buildData(src, gold)
  local srcData = {}
  srcData.words = {}
  srcData.features = {}

  local goldData
  if gold then
    goldData = {}
    goldData.words = {}
    goldData.features = {}
  end

  local ignored = {}
  local indexMap = {}
  local index = 1

  for b = 1, #src do
    if #src[b].words == 0 then
      table.insert(ignored, b)
    else
      indexMap[index] = b
      index = index + 1

      table.insert(srcData.words,
                   self.dicts.src.words:convertToIdx(src[b].words, onmt.Constants.UNK_WORD))

      if #self.dicts.src.features > 0 then
        table.insert(srcData.features,
                     onmt.utils.Features.generateSource(self.dicts.src.features, src[b].features))
      end

      if gold then
        table.insert(goldData.words,
                     self.dicts.tgt.words:convertToIdx(gold[b].words,
                                                       onmt.Constants.UNK_WORD,
                                                       onmt.Constants.BOS_WORD,
                                                       onmt.Constants.EOS_WORD))

        if #self.dicts.tgt.features > 0 then
          table.insert(goldData.features,
                       onmt.utils.Features.generateTarget(self.dicts.tgt.features, gold[b].features))
        end
      end
    end
  end

  return onmt.data.Dataset.new(srcData, goldData), ignored, indexMap
end

function Translator:buildTargetWords(pred, src, attn)
  local tokens = self.dicts.tgt.words:convertToLabels(pred, onmt.Constants.EOS)

  if self.opt.replace_unk then
    for i = 1, #tokens do
      if tokens[i] == onmt.Constants.UNK_WORD then
        local _, maxIndex = attn[i]:max(1)
        local source = src[maxIndex[1]]

        if self.phraseTable and self.phraseTable:contains(source) then
          tokens[i] = self.phraseTable:lookup(source)
        else
          tokens[i] = source
        end
      end
    end
  end

  return tokens
end

function Translator:buildTargetFeatures(predFeats)

  local numFeatures = #predFeats[1]

  if numFeatures == 0 then
    return {}
  end

  local feats = {}
  for _ = 1, numFeatures do
    table.insert(feats, {})
  end

  for i = 2, #predFeats do
    for j = 1, numFeatures do
      table.insert(feats[j], self.dicts.tgt.features[j]:lookup(predFeats[i][j]))
    end
  end

  return feats
end


-- Scores is a table of log-softmax returned by 
-- the decoders 
function Translator:ensembleScore(scores)
	
	
	local score = scores[1]
	
	
	if self.ensembleOps == 'sum' then -- get the average of the probability
		
		score = torch.exp(score) -- so we have to exp to get the prob
		
		for i = 2, self.nModels do
			score:add(torch.exp(scores[i]))
		end
		
		score:div(self.nModels)
		score = torch.log(score)
	else -- logsum operation. 
		
		for i = 2, self.nModels do
			score:add(scores[i])
		end
		
		score:div(self.nModels)
		score = self.logSoftMax:forward(score) 
	end
	
	return score

end

-- For decoder advancer, we have to compute the gold score a bit differently 
function Translator:computeGoldScore(batch, encStates, contexts)


	local goldScore = self.models[1].decoder:computeScore(batch, encStates[1], contexts[1])
	
	for i = 2, self.nModels do
		local score = self.models[i].decoder:computeScore(batch, encStates[i], contexts[i])
		for b = 1, batch.size do
			goldScore[b] = goldScore[b] + score[b]
		end
	end
	
	for b = 1, batch.size do
		goldScore[b] = goldScore[b] / self.nModels
	end


	
	
	--~ local goldScore = self:ensembleScore(scores)
	
	return goldScore
	
	
end

function Translator:translateBatch(batch)

  --~ self.models.encoder:maskPadding()
  --~ self.models.decoder:maskPadding()
  for i = 1, self.nModels do
	self.models[i].encoder:maskPadding()
	self.models[i].decoder:maskPadding()
  end
  
  local encStates = {}
  local contexts = {}
  
  for i = 1, self.nModels do
	encStates[i], contexts[i] = self.models[i].encoder:forward(batch)
  end

  --~ local encStates, context = self.models.encoder:forward(batch)

  -- Compute gold score.
  local goldScore
  if batch.targetInput ~= nil then
    --~ if batch.size > 1 then
      --~ self.models.decoder:maskPadding(batch.sourceSize, batch.sourceLength)
    --~ end
    --~ goldScore = self.models.decoder:computeScore(batch, encStates, context)
    goldScore = self:computeGoldScore(batch, encStates, contexts)
  end
  
  local decoders = {}
  
  for i = 1, self.nModels do
	table.insert(decoders, self.models[i].decoder)
  end

  -- Specify how to go one step forward.
  local advancer = onmt.translate.EnsembleDecoderAdvancer.new(decoders,
                                                      batch,
                                                      contexts,
                                                      self.opt.max_sent_length,
                                                      self.opt.max_num_unks,
                                                      encStates,
                                                      self.dicts, self.opt.word_pen,
                                                      self.ensembleOps)

  -- Save memory by only keeping track of necessary elements in the states.
  -- Attentions are at index 4 in the states defined in onmt.translate.DecoderAdvancer.
  local attnIndex = 4

  -- Features are at index 6 in the states defined in onmt.translate.DecoderAdvancer.
  local featsIndex = 6

  advancer:setKeptStateIndexes({attnIndex, featsIndex})

  -- Conduct beam search.
  local beamSearcher = onmt.translate.BeamSearcher.new(advancer)
  local results = beamSearcher:search(self.opt.beam_size, self.opt.n_best, self.opt.pre_filter_factor)

  local allHyp = {}
  local allFeats = {}
  local allAttn = {}
  local allScores = {}

  for b = 1, batch.size do
    local hypBatch = {}
    local featsBatch = {}
    local attnBatch = {}
    local scoresBatch = {}

    for n = 1, self.opt.n_best do
      local result = results[b][n]
      local tokens = result.tokens
      local score = result.score
      local states = result.states
      local attn = states[attnIndex] or {}
      local feats = states[featsIndex] or {}
      table.remove(tokens)

      -- Remove unnecessary values from the attention vectors.
      local size = batch.sourceSize[b]
      for j = 1, #attn do
		for k = 1, self.nModels do
			attn[j][k] = attn[j][k]:narrow(1, batch.sourceLength - size + 1, size)
		end
      end

      table.insert(hypBatch, tokens)
      if #feats > 0 then
        table.insert(featsBatch, feats)
      end
      table.insert(attnBatch, attn)
      table.insert(scoresBatch, score)
    end

    table.insert(allHyp, hypBatch)
    table.insert(allFeats, featsBatch)
    table.insert(allAttn, attnBatch)
    table.insert(allScores, scoresBatch)
  end

  return allHyp, allFeats, allScores, allAttn, goldScore
end

--[[ Translate a batch of source sequences.

Parameters:

  * `src` - a batch of tables containing:
    - `words`: the table of source words
    - `features`: the table of feaures sequences (`src.features[i][j]` is the value of the ith feature of the jth token)
  * `gold` - gold data to compute confidence score (same format as `src`)

Returns:

  * `results` - a batch of tables containing:
    - `goldScore`: if `gold` was given, this is the confidence score
    - `preds`: an array of `opt.n_best` tables containing:
      - `words`: the table of target words
      - `features`: the table of target features sequences
      - `attention`: the attention vectors of each target word over the source words
      - `score`: the confidence score of the prediction
]]
function Translator:translate(src, gold)
  local data, ignored, indexMap = self:buildData(src, gold)

  local results = {}

  if data:batchCount() > 0 then
    local batch = data:getBatch()

    local pred, predFeats, predScore, attn, goldScore = self:translateBatch(batch)
	
	--~ print(pred, predFeats)
	--~ print(pred[1])
	--~ print(predFeats[1])
    for b = 1, batch.size do
      results[b] = {}

      results[b].preds = {}
      for n = 1, self.opt.n_best do
        results[b].preds[n] = {}
        results[b].preds[n].words = self:buildTargetWords(pred[b][n], src[indexMap[b]].words, attn[b][n])
        results[b].preds[n].features = self:buildTargetFeatures(predFeats[b][n])
        results[b].preds[n].attention = attn[b][n]
        results[b].preds[n].score = predScore[b][n]
      end

      if goldScore ~= nil then
        results[b].goldScore = goldScore[b]
      end
    end
  end

  for i = 1, #ignored do
    table.insert(results, ignored[i], {})
  end

  return results
end

return Translator
