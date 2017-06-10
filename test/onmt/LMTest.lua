require('onmt.init')

local tester = ...

local languageModelTest = torch.TestSuite()

function languageModelTest.basic()
  tester:eq(LanguageModel.dataType('monotext'), true)
  tester:eq(LanguageModel.dataType(), 'monotext')
end

function languageModelTest.train()
  local cmd = onmt.utils.ExtendedCmdLine.new()
  LanguageModel.declareOpts(cmd)
  local args = cmd:parse({'-word_vec_size', '10', '-layers', '1', '-rnn_size', '10'})
  local d = onmt.utils.Dict.new()
  d:add('How')
  d:add('are')
  d:add('you')
  d:add('?')
  local dicts = { src={ words=d, features={} } }
  local lm = LanguageModel.new(args, dicts)
  local text = { 'How', 'are', 'you', '?' }
  local words = onmt.utils.Features.extract(text)
  local wordsIdx = dicts.src.words:convertToIdx(words)
  local dataset = onmt.data.Dataset.new({ words={ wordsIdx }, features={} })
  local batch = dataset:getBatch(1)
  lm:training()
  lm:trainNetwork(batch)
  lm:evaluate()
  local loss = lm:forwardComputeLoss(batch)
  tester:assertgt(loss, 0)
  local indvLoss
  loss, indvLoss = lm:forwardComputeLoss(batch, true)
  tester:eq(indvLoss:size(1), 1)
end

return languageModelTest
