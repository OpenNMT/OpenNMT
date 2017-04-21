--[[ BLEU is a class used for compute sentence-level BLEU score --]]
local threads = require('threads')

local BLEU = {
  n = 4,
}

-- Notation: the BLEU score here is not comparable with the one 
-- computed by external tools, since 
-- - we use smoothed BLEU score for sentence level
-- - we also reward BLEU score for the end of sentence token. 
--  Removing that token can result in significant decrease in BLEU (maybe comparable with external scripts)

function BLEU.resetStats(corpusStats)
	corpusStats.hypNgrams = {}
	corpusStats.refNgrams = {}
	corpusStats.nSentences = 0
	corpusStats.hypLength = 0
	corpusStats.refLength = 0
end


-- simply insert the ngram into a stat table
function BLEU.insertEntry(record, ngram, len)
	
	if record[ngram] == nil then
		record[ngram] = {1, len}
	else
		record[ngram][1] = record[ngram][1] + 1
	end
	
end

-- Function to get ngram precision from 2 sentences
-- Hyp and Ref are both in table of splitted words form
function BLEU.getNgramPrecision(hyp, ref)
	
	local results = {}
	
	for i = 1, BLEU.n do
		results[i] = {0, 0} -- total, correct
	end
	
	-- extract the ngrams 
	local hyp_ngrams, hyp_len = BLEU.getNgrams(hyp)
	
	local ref_ngrams, ref_len = BLEU.getNgrams(ref)
	
	
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

-- extract n-grams from a table of splitted words
function BLEU.getNgrams(sentence)
	length = #sentence
	
	local ngrams = {}
	
	for i = 1, length do
		for j = i, math.min(i + BLEU.n - 1, length) do
			local ngramLen = j - i + 1
			local ngram = ""
			for k = i, j do
				ngram = ngram .. sentence[k] .. " "
			end
			BLEU.insertEntry(ngrams, ngram, ngramLen)
		end
	end
	
	return ngrams, length
end



-- Assuming hyp and ref are tables of words already preprocessed by Rewarder
-- Note: they don't start with <s> (BOS)
function BLEU.computeScore(hyp, ref)	
	
	local r, hyp_len, ref_len = BLEU.getNgramPrecision(hyp, ref)
	
	-- brevity penalty
	local bp = math.exp(1 - math.max(1, ref_len / hyp_len))
	
	local correct = 0
	local total = 0
		
	local score = torch.Tensor(BLEU.n)
	
	local bleu = 1.
	
	for i = 1, BLEU.n do
		
		if r[i][1] > 0 then
			--~ if r[i][2] == 0 then
				--~ m = m * 0.5 -- smooth bleu score
				--~ r[i][2] = m
			--~ end
			
			local prec = (r[i][2] + 1) / (r[i][1] + 1) -- smooth bleu score
 			bleu = bleu * prec
 			--~ score[i] = prec
		end
	end
	
	--~ local bleu = score:log():sum(1):div(self.n):exp():squeeze()
	
	bleu = bleu ^ ( 1 / BLEU.n )
	
	bleu = bleu * bp 
	
	return bleu
end

-- This is sentence BLEU score for each sentence pair in the minibatch 
-- size: length * batch (vectorised form)
--~ function BLEU:computeScore(hypBatch, refBatch)

	
--~ end


-- Accumulate the scores 
function BLEU.accumulateCorpusScore(corpusStats, hyp, ref)
	
	local hyp_ngrams, hyp_len = BLEU.getNgrams(hyp)

	
	-- accumulate stats for hyp ngrams
	for ngram, d in pairs(hyp_ngrams) do
		local count = d[1]
		local len = d[2]
				
		BLEU.insertEntry(corpusStats.hypNgrams, ngram, len)
	end
	
	local ref_ngrams, ref_len = BLEU.getNgrams(ref)
	
	-- accumulate stats for ref ngrams
	for ngram, d in pairs(ref_ngrams) do
		local count = d[1]
		local len = d[2]
	
		
		BLEU.insertEntry(corpusStats.refNgrams, ngram, len)
	end
	
	-- accumulate length 
	corpusStats.hypLength = corpusStats.hypLength + hyp_len
	corpusStats.refLength = corpusStats.refLength + ref_len
end




--~ -- should call this function after accumulate all
function BLEU.getNgramPrecisionCorpus(corpusStats)
	local results = {}
	
	
	for i = 1, BLEU.n do
		results[i] = {0, 0} -- total, correct
	end
	
	-- extract the ngrams 
	local hyp_ngrams = corpusStats.hypNgrams
	local ref_ngrams = corpusStats.refNgrams
	
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

function BLEU.computeCorpusScore(corpusStats)
	
	local r = BLEU.getNgramPrecisionCorpus(corpusStats)
	
	local hyp_len = corpusStats.hypLength
	local ref_len = corpusStats.refLength
		
	-- brevity penalty
	local bp = math.exp(1 - math.max(1, ref_len / hyp_len))
	
	local correct = 0
	local total = 0
		
	local score = torch.Tensor(BLEU.n)
	
	for i = 1, BLEU.n do
		
		if r[i][1] > 0 then
			
			local prec = r[i][2] / r[i][1] -- For corpus level we don't smooth
			score[i] = prec
		end
	end
		
	local bleu = score:log():sum(1):div(BLEU.n):exp():squeeze()
	
	bleu = bleu * bp * 100
	
	return bleu, bp
	
end

return BLEU

