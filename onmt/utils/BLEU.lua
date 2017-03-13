--[[ BLEU is a class used for compute sentence-level BLEU score 
--]]
local BLEU = torch.class('BLEU')


function BLEU:__init(n, count)
	self.score = 0
	self.ngrams = 0
	self.n = n or 4 -- maximum number of ngrams to consider
	self.count = count or 1
end

-- Function to get ngram precision from 2 sentences
-- Hyp and Ref are both in vectorised form
function BLEU:getNgramPrecision(hyp, ref)
	
	local results = {}
	
	for i = 1, n do
		results[i] = {0, 0} -- total, correct
	end
	
	-- extract the ngrams 
	local hyp_ngrams = self:getNgrams(hyp)
	local ref_ngrams = self:getNgrams(ref)
	
	for ngram, d in pairs(hyp_ngrams) do
		
		local count = d[1]
		local len = d[2]
		
		results[len][1] = results[len][1] + count
		
		local actual
		if ref_ngrams[ngram] == nil then
			actual = 0
		else
			actual = ref_ngrams[ngram][1]
		end
		
		results[len][2] = results[len][2] + math.min(actual, count) 
	end
	
	return results
end


-- A function to get ngram statistics from a sentence
-- Input: sentence (vector of idx)
function BLEU:getNgrams(sentence)
	
	local ngrams = {}
	local length = sentence:size(1)
	
	for i = 1, length do
		for j = math.min(i + self.n - 1, length) do
			local ngramLen = j - i + 1
			local ngram = sentence:narrow(1, i, ngramLen)
			if self.count == 0 then
				table.insert(ngrams, ngram)
			else
				if ngrams[ngram] == nil then
					ngrams[ngram] = {1, ngramLen}
				else
					ngrams[ngram][1] = ngrams[ngram][1] + 1
				end
			end
		end
	end
	
	return ngrams
end

-- Assuming hyp and ref are 1-dim tensors with length l_h and l_r
function BLEU:computeBLEU(hyp, ref)
	
	local m = 1
	
	local r = self:getNgramPrecision(hyp, ref)
	local hyp_len = hyp:size(1)
	local ref_len = ref:size(1)
	local bp = math.exp(1 - math.max(1, ref_len / hyp_len))
	
	local correct = 0
	local total = 0
	
	local bleu = 1
	
	for i = 1, n do
		
		if r[i][1] > 0 then
			if r[i][2] == 0 then
				m = m * 0.5
				r[i][2] = m
			end
			
			local prec = r[i][2] / r[i][1]
			bleu = bleu * prec
		end
	end
	
	bleu = bleu ^ ( 1 / n )
	
	bleu = bleu * bp 
	
	return bleu
end
