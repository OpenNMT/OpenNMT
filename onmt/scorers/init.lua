local scorers = {}

scorers.bleu = require 'onmt.scorers.bleu'
scorers.ter = require 'onmt.scorers.tercom'
scorers.dlratio = require 'onmt.scorers.dlratio'

-- Build list of available scorers.
scorers.list = {}
for k, _ in pairs(scorers) do
  table.insert(scorers.list, k)
end

-- Mark here scorers that support multiple references.
scorers.multi = {}
scorers.multi['bleu'] = true
scorers.multi['ter'] = true

return scorers
