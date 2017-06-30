local scorers = {}

scorers.bleu = require 'onmt.scorers.bleu'
scorers.dlratio = require 'onmt.scorers.dlratio'

-- Build list of available scorers.
scorers.list = {}
for k, _ in pairs(scorers) do
  table.insert(scorers.list, k)
end

-- Mark here scorers that support multiple references.
scorers.multi = {}
scorers.multi['bleu'] = true

return scorers
