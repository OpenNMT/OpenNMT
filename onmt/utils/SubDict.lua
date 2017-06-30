--[[Read subdictionary file and builds mapping to complete dictionary
--]]
local SubDict = torch.class('SubDict')

function SubDict:__init(dict, filePath)
  local f = assert(io.open(filePath, 'r'))

  self.vocabs = { onmt.Constants.UNK, onmt.Constants.BOS, onmt.Constants.EOS }

  for line in f:lines() do
    local idx = dict:lookup(onmt.utils.String.strip(line))
    if idx then
      table.insert(self.vocabs, idx)
    end
  end
  f:close()
  self.targetVocTensor, self.targetVocInvMap = torch.LongTensor(self.vocabs):sort()
  for i = 1, self.targetVocInvMap:size(1) do
    self.targetVocInvMap[i] = self.vocabs[self.targetVocInvMap[i]]
  end
  self.vocabs = nil
end

function SubDict:fullIdx(tensor)
  if tensor.map then
      tensor:map(tensor, function(idx) return self.targetVocInvMap[idx] end)
  else
    for idx = 1, tensor:size(1) do
      tensor[idx] = self.targetVocInvMap[tensor[idx]]
    end
  end
  return
end

return SubDict
