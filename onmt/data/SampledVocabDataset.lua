local SampledVocabDataset, parent = torch.class("SampledVocabDataset", "Dataset")

local options = {
  {
    '-sample_tgt_vocab', false,
    [[Use importance sampling as an approximation of the full output vocabulary softmax.]],
    {
      deprecatedBy = { 'sample_vocab', 'true' }
    }
  },
  {
    '-sample_vocab', false,
    [[Use importance sampling as an approximation of the full output vocabulary softmax.]],
    {
      depends = function(opt)
                  if opt.sample_vocab then
                    if opt.model_type and opt.model_type ~= 'seq2seq' and opt.model_type ~= 'lm' then
                      return false, "only works for seq2seq or language models." end
                    if opt.sample == 0 and opt.gsample == 0 then return false, "requires '-sample' or '-gsample' option" end
                  end
                  return true
                end
    }
  }
}

function SampledVocabDataset.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Sampled Vocabulary')
end

--[[ Initialize a data object given aligned tables of IntTensors `srcData`
  and `tgtData`.
--]]
function SampledVocabDataset:__init(srcData, tgtData)
  parent.__init(self, srcData, tgtData)
end

--[[ Methods used for sampled SampledVocabDatasets ]]
function SampledVocabDataset:sampleVocabInit(opt, src, tgt)
  if opt.sample_vocab then
    self.vocabIndex = {}
    self.vocabTensor = {}
    if opt.model_type == 'lm' then
      self.vocabAxis = src
      self.vocabAxisName = 'source'
    else
      self.vocabAxis = tgt
      self.vocabAxisName = 'target'
    end
    if self.vocabIndex then
      _G.logger:info(' * with ' .. self.vocabAxisName .. ' vocabulary importance sampling')
    end
  end
end

function SampledVocabDataset:sampleVocabLog()
end

function SampledVocabDataset:sampleVocabClear()
  if self.vocabIndex then
    self.vocabTensor = {}
    self.vocabIndex = {}
    self.vocabIndex[onmt.Constants.PAD] = 1
    table.insert(self.vocabTensor, onmt.Constants.PAD)
  end
end

function SampledVocabDataset:selectVocabs(sampled)
  if self.vocabIndex then
    for j = 1, self.vocabAxis[sampled]:size(1) do
      if not self.vocabIndex[self.vocabAxis[sampled][j]] then
        self.vocabIndex[self.vocabAxis[sampled][j]] = 1
        table.insert(self.vocabTensor, self.vocabAxis[sampled][j])
      end
    end
  end
end

function SampledVocabDataset:sampleVocabReport(logLevel)
  if self.vocabIndex then
    self.vocabTensor = torch.LongTensor(self.vocabTensor):sort()
    _G.logger:log('Importance Sampling - keeping ' .. self.vocabTensor:size(1) .. ' ' .. self.vocabAxisName .. ' vocabs.', logLevel)
  end
end

function SampledVocabDataset:sampleVocabIdx(dir, idx)
  if self.vocabAxisName == dir then
    return onmt.utils.Tensor.find(self.vocabTensor, idx)
  end
  return idx
end

return SampledVocabDataset
