require('onmt.init')

local tester = ...

local decoderTest = torch.TestSuite()

local criterion = onmt.ParallelClassNLLCriterion({10})

local function buildDecoder(inputFeed, rnnType, layers)
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.Encoder.declareOpts(cmd)
  onmt.GlobalAttention.declareOpts(cmd)

  local opt = cmd:parse('')
  opt.rnn_size = 10
  opt.rnn_type = rnnType or 'LSTM'
  opt.dropout = 0
  opt.input_feed = inputFeed and 1 or 0
  opt.layers = layers or opt.layers

  local inputNet = nn.LookupTable(10, 4)
  inputNet.inputSize = 4

  local generator = onmt.Generator(opt, {10})
  local attention = onmt.GlobalAttention(opt, opt.rnn_size)

  return onmt.Decoder(opt, inputNet, generator, attention), opt
end

local function generateEncoderStates(batchSize, timesteps, opt)
  local numStates = opt.rnn_type == 'LSTM' and 2 or 1

  local context = torch.Tensor(batchSize, timesteps, opt.rnn_size):uniform(-0.1, 0.1)
  local states = {}
  for _ = 1, numStates * opt.layers do
    table.insert(states, torch.Tensor(batchSize, opt.rnn_size):uniform(-0.1, 0.1))
  end

  return { states, context }
end

local function checkDim(decoder, opt, encoderStates, withIndLosses)
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6}),
    torch.IntTensor({5, 6, 7}),
  }
  local tgt = {
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, 9, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, onmt.Constants.EOS}),
    torch.IntTensor({onmt.Constants.BOS, 5, 6, 7, 8, onmt.Constants.EOS}),
  }

  local batch = onmt.data.Batch.new(src, {}, tgt, {})

  decoder:training()

  if not encoderStates then
    encoderStates = generateEncoderStates(3, 4, opt)
  end

  local outputs = decoder:forward(batch, encoderStates[1], encoderStates[2])

  tester:eq(#outputs, 6)
  tester:eq(outputs[1]:size(), torch.LongStorage({3, opt.rnn_size}))

  if withIndLosses then
    decoder:returnIndividualLosses(true)
  end

  local gradStates, gradContext, loss, indLosses = decoder:backward(batch, outputs, criterion)

  tester:eq(#gradStates, 1 + opt.layers * (opt.rnn_type == 'GRU' and 1 or 2))
  tester:eq(gradContext:size(), encoderStates[2]:size())
  tester:eq(type(loss), 'number')

  if withIndLosses then
    tester:eq(indLosses:size(1), 3)
    tester:eq(indLosses:cmul(batch.targetSize:double()):sum(), loss, 1e-8)
  end

  return outputs, encoderStates
end

local function checkSerial(decoder, opt)
  local outputs, encoderStates = checkDim(decoder, opt)

  torch.save('dec.t7', decoder:serialize())
  local reload = decoder.load(torch.load('dec.t7'))
  os.remove('dec.t7')

  local newOutputs = checkDim(reload, opt, encoderStates)

  tester:eq(newOutputs, outputs)
end

local function moduleExists(obj, typename)
  local exists = false
  obj:apply(function (m)
    if torch.typename(m) == typename then
      exists = true
    end
  end)
  return exists
end

function decoderTest.simple_LSTM()
  local decoder, opt = buildDecoder(true, 'LSTM')
  checkDim(decoder, opt)
end

function decoderTest.saveAndLoad_LSTM()
  local decoder, opt = buildDecoder(true, 'LSTM')
  checkSerial(decoder, opt)
end

function decoderTest.withoutInputFeeding_LSTM()
  local decoder, opt = buildDecoder(false, 'LSTM')
  checkDim(decoder, opt)
end

function decoderTest.simple_GRU()
  local decoder, opt = buildDecoder(true, 'GRU')
  checkDim(decoder, opt)
end

function decoderTest.simple_GRU_oneLayer()
  local decoder, opt = buildDecoder(true, 'GRU', 1)
  checkDim(decoder, opt)
end

function decoderTest.saveAndLoad_GRU()
  local decoder, opt = buildDecoder(true, 'GRU')
  checkSerial(decoder, opt)
end

function decoderTest.withoutInputFeeding_GRU()
  local decoder, opt = buildDecoder(false, 'GRU')
  checkDim(decoder, opt)
end

function decoderTest.masking()
  local decoder, _ = buildDecoder(false, 'LSTM')

  decoder:evaluate()

  decoder:maskPadding(torch.LongTensor({2,5,3,5}), 5)
  tester:eq(moduleExists(decoder, 'onmt.MaskedSoftmax'), true)

  decoder:maskPadding()
  tester:eq(moduleExists(decoder, 'onmt.MaskedSoftmax'), false)
end

function decoderTest.individualLosses()
  local decoder, opt = buildDecoder(true, 'LSTM')
  checkDim(decoder, opt, nil, true)
end

return decoderTest
