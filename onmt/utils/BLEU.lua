--[[ BLEU is a class used for compute sentence-level BLEU score 
--]]

local pl = require('pl.import_into')()
local BLEU = torch.class('BLEU')

-- Notation: the BLEU score here is not comparable with the one 
-- computed by external tools, since 
-- - we use smoothed BLEU score for sentence level
-- - we also reward BLEU score for the end of sentence token. 
--  Removing that token can result in significant decrease in BLEU (maybe comparable with external scripts)
function BLEU:__init(dict, n, count, bpe)
	
	self.dict = dict
	self.ngrams = 0
	self.n = n or 4 -- maximum number of ngrams to consider
	--~ self.count = count or 1
	self.countRule = count or 1
	self.eos = onmt.Constants.EOS
	self.pad = onmt.Constants.PAD
	
	self.corpusStats = {} 
	
	self.corpusStats.hypNgrams = {}
	self.corpusStats.refNgrams = {}
	self.corpusStats.nSentences = 0
	self.corpusStats.hypLength = 0
	self.corpusStats.refLength = 0
	self.bpe = bpe or true
end

function BLEU:resetStats()
	self.corpusStats.hypNgrams = {}
	self.corpusStats.refNgrams = {}
	self.corpusStats.nSentences = 0
	self.corpusStats.hypLength = 0
	self.corpusStats.refLength = 0
end

-- convert the vector to sentence
function BLEU:getSentence(ids)
	
	local labels, sentence = self.dict:convertToLabels(ids, onmt.Constants.EOS, true)
	local length = #labels
	
	return sentence, length
end

local function getLength(object)
	
	local length = #object
	
	if torch.type(length) == 'torch.LongStorage' then
		length = length[1] end
	
	return length
end

-- find the real length of this sent-vector (with PAD and EOS)
function BLEU:getRealLength(sentence)

	local length = getLength(sentence)
	
	
	local realLength = length
	
	for i = 1, length do
		if sentence[i] == onmt.Constants.EOS then
			realLength = i
			break
		end
	end
	return realLength
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


--~ -- simply insert the ngram into a table 
local function insertEntry(record, ngram, len, count)
	count = count or 1
	if count == 0 then
		table.insert(record, ngram)
	else 
		if record[ngram] == nil then
			record[ngram] = {1, len}
		else
			record[ngram][1] = record[ngram][1] + 1
		end
	end
end

-- Function to get ngram precision from 2 sentences
-- Hyp and Ref are both in vectorised form
function BLEU:getNgramPrecision(hyp, ref)
	
	local results = {}
	
	for i = 1, self.n do
		results[i] = {0, 0} -- total, correct
	end
	
	-- extract the ngrams 
	local hyp_ngrams, hyp_len = self:getNgrams(hyp)
	
	local ref_ngrams, ref_len = self:getNgrams(ref)
	
	
	--~ local ngram
	for ng, d in pairs(hyp_ngrams) do
	
		
		local count = d[1]
		local len = d[2]
		
		results[len][1] = results[len][1] + count
		
		local actual
		
		
		if ref_ngrams[ng] == nil then
			actual = 0
		else
			actual = ref_ngrams[ng][1]
		end
		
		results[len][2] = results[len][2] + math.min(actual, count) 
	end
	
	return results, hyp_len, ref_len
end


-- A function to get ngram statistics from a sentence
-- Input: sentence (vector of idx)
-- The vector SHOULD NOT start with <BOS>, but the <EOS> and <PAD> can exist
-- Output: a table of ngrams (converted to string), with the real length of the sentence (even after BPE)
function BLEU:getNgrams(ids)
	
	assert(torch.isTensor(ids) == true, 'expect a vector of ids here')

	local sentence, length = self:getSentence(ids)
	
	local processed 
	if self.bpe == true then
		processed = deBPE(sentence)
	else
		processed = sentence
	end
	
	-- we have to split the sentence again
	sentence = split(processed)
	
	-- reupdate the length (after BPE the length can be different)
	length = #sentence
	
	local ngrams = {}
		
	for i = 1, length do
		for j = i, math.min(i + self.n - 1, length) do
			local ngramLen = j - i + 1
			local ngram = ""
			for k = i, j do
				ngram = ngram .. sentence[k] .. " "
			end
			insertEntry(ngrams, ngram, ngramLen, self.countRule)
		end
	end
	
	return ngrams, length
end

-- Assuming hyp and ref are 1-dim tensors with length l_h and l_r
-- Note: they don't start with <s> (BOS)
function BLEU:computeBLEU(hyp, ref)

	
	local m = 1
	
	local r, hyp_len, ref_len = self:getNgramPrecision(hyp, ref)
	
	-- brevity penalty
	local bp = math.exp(1 - math.max(1, ref_len / hyp_len))
	
	local correct = 0
	local total = 0
	
	local bleu = 1
	
	for i = 1, self.n do
		
		if r[i][1] > 0 then
			if r[i][2] == 0 then
				m = m * 0.5 -- smooth bleu score
				r[i][2] = m
			end
			
			local prec = r[i][2] / r[i][1]
			bleu = bleu * prec
		end
	end
	
	bleu = bleu ^ ( 1 / self.n )
	
	bleu = bleu * bp 
	
	return bleu
end

-- This is sentence BLEU score for each sentence pair in the minibatch 
-- size: length * batch (vectorised form)
function BLEU:computeScore(hypBatch, refBatch)

	refBatch = refBatch:cuda()
	local batchSize = hypBatch:size(2)
	
	local bleuScore = torch.Tensor(batchSize)
	
	for b = 1, batchSize do
		
		bleuScore[b] = self:computeBLEU(hypBatch[{{},b}], refBatch[{{},b}])
	end
	
	return bleuScore
end

--~ function BLEU:accumulate

-- Accumulate the scores 
function BLEU:accumulateCorpusScore(hyp, ref)
	
	local hyp_ngrams, hyp_len = self:getNgrams(hyp)

	
	-- accumulate stats for hyp ngrams
	for ngram, d in pairs(hyp_ngrams) do
		local count = d[1]
		local len = d[2]
				
		insertEntry(self.corpusStats.hypNgrams, ngram, len)
	end
	
	local ref_ngrams, ref_len = self:getNgrams(ref)
	
	-- accumulate stats for ref ngrams
	for ngram, d in pairs(ref_ngrams) do
		local count = d[1]
		local len = d[2]
	
		
		insertEntry(self.corpusStats.refNgrams, ngram, len)
	end
	
	-- accumulate length 
	self.corpusStats.hypLength = self.corpusStats.hypLength + hyp_len
	self.corpusStats.refLength = self.corpusStats.refLength + ref_len
end

--~ 
function BLEU:accumulateCorpusScoreBatch(hypBatch, refBatch)
	
	local batchSize = hypBatch:size(2)
	
	assert(batchSize == refBatch:size(2))
	
	for b = 1, batchSize do
		self:accumulateCorpusScore(hypBatch[{{},b}], refBatch[{{},b}])
	end
end

-- should call this function after accumulate all
function BLEU:getNgramPrecisionCorpus()
	local results = {}
	
	
	for i = 1, self.n do
		results[i] = {0, 0} -- total, correct
	end
	
	-- extract the ngrams 
	local hyp_ngrams = self.corpusStats.hypNgrams
	local ref_ngrams = self.corpusStats.refNgrams
	
	for ng, d in pairs(hyp_ngrams) do
		local count = d[1]
		local len = d[2]
		
		results[len][1] = results[len][1] + count
		
		local actual
		if ref_ngrams[ng] == nil then
			actual = 0
		else
			actual = ref_ngrams[ng][1]
		end
		
		results[len][2] = results[len][2] + math.min(actual, count) 
	end
	
	return results
end

function BLEU:computeCorpusBLEU()
	local m = 1
	
	local r = self:getNgramPrecisionCorpus()
	
	local hyp_len = self.corpusStats.hypLength
	local ref_len = self.corpusStats.refLength
		
	-- brevity penalty
	local bp = math.exp(1 - math.max(1, ref_len / hyp_len))
	
	local correct = 0
	local total = 0
	
	local bleu = 1
	
	for i = 1, self.n do
		
		if r[i][1] > 0 then
			if r[i][2] == 0 then
				m = m * 0 -- smooth bleu score. For corpus level we don't smooth
				r[i][2] = m
			end
			
			local prec = r[i][2] / r[i][1]
			bleu = bleu * prec
		end
	end
	
	bleu = bleu ^ ( 1 / self.n )
	
	bleu = bleu * bp * 100
	
	return bleu, bp
	
end

return BLEU

