
--[[ Rewarder is a class dedicated for hand-crafted score --]]
local threads = require('threads')

local Rewarder = torch.class('Rewarder')
local pl = require('pl.import_into')()


function Rewarder:__init(dict, bpe, reward_type)
	
	self.dict = dict -- the dictionary from the data (target side)
	self.bpe = bpe or true -- remove bpe to get a better 'approximation' of BLEU
	
	self.reward_type = reward_type or 'bleu'
	
	if self.reward_type == 'bleu' then
		self.scorer = require('onmt.utils.BLEU')
	end
		
	self.sentenceScore = onmt.utils.Cuda.convert(torch.Tensor())
	
	self.corpusStats = {}
	self:resetCorpusStats()
end

-- for corpus-level metrics
function Rewarder:resetCorpusStats()
	self.scorer.resetStats(self.corpusStats)
end

-- an util function to get length of a table or a vector
local function getLength(object)
	
	local length = #object
	
	if torch.type(length) == 'torch.LongStorage' then
		length = length[1] end
	
	return length
end

-- generate a list of words for a vector/list of ids
local function getSentence(ids, dict)
	
	local labels, sentence = dict:convertToLabels(ids, onmt.Constants.EOS, true)
	local length = #labels
	
	return sentence
end

-- A function to get rid of BPE. Simply remove the @@ away. 
local function deBPE(sentence)
	local newSentence = sentence:gsub("@@ ", "")
	return newSentence
end

-- Split a string into table of words
local function split(sentence)
	local words = pl.utils.split(sentence, ' ')
	return words
end

-- a fully wrapped-up function to restore the sentence given the ids
local function ids2words(ids, dict)
	
	local sentence = getSentence(ids, dict) -- convert to words
	local preprocessed = deBPE(sentence) -- normalize BPE
	local words = split(preprocessed) -- then split to words again
	
	return words
end

-- Compute score for each sentence pair in the minibatch 
-- size: length * batchSize (vectorised form)
-- return a tensor of size batchSize
function Rewarder:computeScore(hypBatch, refBatch)
	
	-- float can be faster for indexing ?
	hypBatch = hypBatch:float()
	refBatch = refBatch:float()
	local batchSize = hypBatch:size(2)
	
	local hypTable = {}
	local refTable = {}
	
	self.sentenceScore:resize(batchSize):zero()
	
	-- build list of words for each minibatch
	for b = 1, batchSize do
		hypTable[b] = ids2words(hypBatch[{{},b}], self.dict)
		refTable[b] = ids2words(refBatch[{{},b}], self.dict)
		
		self.sentenceScore[b] = self.scorer.computeScore(hypTable[b], refTable[b])
	end
	
	return self.sentenceScore
end


function Rewarder:accumulateCorpusScoreBatch(hypBatch, refBatch)
	
	hypBatch = hypBatch:float()
	refBatch = refBatch:float()
	
	local batchSize = hypBatch:size(2)
	
	-- accumulate score for each sentence
	for b = 1, batchSize do
		self:accumulateCorpusScore(hypBatch[{{},b}], refBatch[{{},b}])
	end
end

function Rewarder:accumulateCorpusScore(hyp, ref)
	-- convert vectors to words
	local hypSent = ids2words(hyp, self.dict)
	local refSent = ids2words(ref, self.dict)
	
	self.scorer.accumulateCorpusScore(self.corpusStats, hypSent, refSent) 
end

function Rewarder:computeCorpusScore()
	self.corpusScore = self.scorer.computeCorpusScore(self.corpusStats)
	return self.corpusScore
end
