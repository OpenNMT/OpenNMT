require('onmt.init')

local tester = ...

local encoderTest = torch.TestSuite()

local function buildEncoder(class, rnnType, merge, dropout, dropout_type, dropout_input)
  local cmd = onmt.utils.ExtendedCmdLine.new()
  class.declareOpts(cmd)

  local opt = cmd:parse('')
  opt.rnn_size = 10
  opt.rnn_type = rnnType or 'LSTM'
  opt.brnn_merge = merge or 'sum'
  opt.dropout = dropout or 0
  opt.dropout_type = dropout_type or 'naive'
  opt.dropout_input = dropout_input or false
  opt.max_batch_size = 64

  if class == onmt.CNNEncoder then
    opt.cnn_size = 10
  end

  local inputNet = nn.LookupTable(10, 4)
  inputNet.inputSize = 4

  return class(opt, inputNet), opt
end

local function genericCheckDim(encoder, opt)
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
  }

  local batch = onmt.data.Batch.new(src)

  encoder:training()

  local states, context = encoder:forward(batch)

  local timesteps = 4
  if opt.pdbrnn_reduction then
    timesteps = timesteps / (opt.pdbrnn_reduction * (opt.layers - 1))
  end

  tester:eq(context:size(), torch.LongStorage({3, timesteps, opt.rnn_size}))
  if not opt.cnn_size then
    tester:eq(#states, opt.layers * (opt.rnn_type == 'LSTM' and 2 or 1))
  end
  for _, v in ipairs(states) do
    tester:eq(v:size(), torch.LongStorage({3, opt.rnn_size}))
  end

  local gradContextOutput = context:clone():uniform(-0.1, 0.1)
  local gradStatesOutput = {}
  for _, v in ipairs(states) do
    table.insert(gradStatesOutput, v:clone():uniform(-0.1, 0.1))
  end

  local gradInputs = encoder:backward(batch, gradStatesOutput, gradContextOutput)

  local steps = #gradInputs
  if opt.cnn_size then
    steps = gradInputs:size(2)
  end
  tester:eq(steps, 4)

  if not opt.cnn_size then
    for _, v in ipairs(gradInputs) do
      tester:eq(v:size(), torch.LongStorage({3}))
    end
  else
    tester:eq(gradInputs:size(1), 3)
  end

  return states, context
end

local function justForward(encoder)
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6, 7, 8}),
  }
  local batch = onmt.data.Batch.new(src)
  encoder:training()
  encoder:forward(batch)
end

local function justForwardOne(encoder, value, clone)
  local inputs =torch.IntTensor(3):fill(value)
  encoder:training()
  return encoder:forwardOne(inputs, nil, clone)
end

local function genericCheckMasking(encoder)
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6}),
    torch.IntTensor({5, 6, 7}),
  }

  local batch = onmt.data.Batch.new(src)

  local _, context = encoder:forward(batch)

  tester:eq(context[2][1]:ne(0):sum(), 0)
  tester:eq(context[2][2]:ne(0):sum(), 0)
  tester:eq(context[3][1]:ne(0):sum(), 0)
end

local function genericCheckSerial(encoder, opt)
  local states, context = genericCheckDim(encoder, opt)

  torch.save('enc.t7', encoder:serialize())
  local reload = encoder.load(torch.load('enc.t7'))
  os.remove('enc.t7')

  local newStates, newContext = genericCheckDim(reload, opt)

  tester:eq(newStates, states)
  tester:eq(newContext, context)
end

function encoderTest.simple_LSTM()
  local encoder, opt = buildEncoder(onmt.Encoder, 'LSTM')
  genericCheckDim(encoder, opt)
end

function encoderTest.simple_LSTM_dropouts()
  local dropout_updateOutput= nn.Dropout.updateOutput
  local count
  nn.Dropout.updateOutput = function(...) count=count+1;return dropout_updateOutput(...) end
  local vdropout_updateOutput= onmt.VariationalDropout.updateOutput
  local vcount
  onmt.VariationalDropout.updateOutput = function(...) vcount=vcount+1;return vdropout_updateOutput(...) end

  count = 0
  vcount = 0
  local encoder = buildEncoder(onmt.Encoder, 'LSTM', 'sum', 0)
  justForward(encoder)
  tester:eq(count, 4)
  tester:eq(vcount, 0)

  count = 0
  encoder = buildEncoder(onmt.Encoder, 'LSTM', 'sum', 0.2)
  justForward(encoder)
  tester:eq(count, 4)
  tester:eq(vcount, 0)

  local context1, context2
  context1 = select(2, justForwardOne(encoder, 2, false))
  context2 = select(2, justForwardOne(encoder, 3, false))
  tester:assertTensorEq(context1, context2)
  context1 = select(2, justForwardOne(encoder, 2, true))
  context2 = select(2, justForwardOne(encoder, 3, true))
  tester:assertTensorNe(context1, context2)

  count = 0
  encoder = buildEncoder(onmt.Encoder, 'LSTM', 'sum', 0.2, 'variational')
  justForward(encoder)
  tester:eq(count, 0)
  tester:eq(vcount, 12)

  vcount = 0
  encoder = buildEncoder(onmt.Encoder, 'LSTM', 'sum', 0.2, 'naive', true)
  justForward(encoder)
  tester:eq(count, 8)
  tester:eq(vcount, 0)

  count = 0
  encoder = buildEncoder(onmt.Encoder, 'LSTM', 'sum', 0.2, 'variational', true)
  justForward(encoder)
  tester:eq(count, 0)
  tester:eq(vcount, 16)

  nn.Dropout.updateOutput = dropout_updateOutput
  onmt.VariationalDropout.updateOutput = vdropout_updateOutput
end

function encoderTest.simple_masking_LSTM()
  local encoder, _ = buildEncoder(onmt.Encoder, 'LSTM')
  genericCheckMasking(encoder)
end

function encoderTest.simple_saveAndLoad_LSTM()
  local encoder, opt = buildEncoder(onmt.Encoder, 'LSTM')
  genericCheckSerial(encoder, opt)
end

function encoderTest.simple_GRU()
  local encoder, opt = buildEncoder(onmt.Encoder, 'GRU')
  genericCheckDim(encoder, opt)
end

function encoderTest.simple_masking_GRU()
  local encoder, _ = buildEncoder(onmt.Encoder, 'GRU')
  genericCheckMasking(encoder)
end

function encoderTest.simple_saveAndLoad_GRU()
  local encoder, opt = buildEncoder(onmt.Encoder, 'GRU')
  genericCheckSerial(encoder, opt)
end

function encoderTest.brnn_LSTM()
  local encoder, opt = buildEncoder(onmt.BiEncoder, 'LSTM')
  genericCheckDim(encoder, opt)
end

function encoderTest.brnn_concat_LSTM()
  local encoder, opt = buildEncoder(onmt.BiEncoder, 'LSTM', 'concat')
  genericCheckDim(encoder, opt)
end

function encoderTest.brnn_masking_LSTM()
  local encoder, _ = buildEncoder(onmt.BiEncoder, 'LSTM')
  genericCheckMasking(encoder)
end

function encoderTest.brnn_saveAndLoad_LSTM()
  local encoder, opt = buildEncoder(onmt.BiEncoder, 'LSTM')
  genericCheckSerial(encoder, opt)
end

function encoderTest.brnn_GRU()
  local encoder, opt = buildEncoder(onmt.BiEncoder, 'GRU')
  genericCheckDim(encoder, opt)
end

function encoderTest.brnn_masking_GRU()
  local encoder, _ = buildEncoder(onmt.BiEncoder, 'GRU')
  genericCheckMasking(encoder)
end

function encoderTest.brnn_saveAndLoad_GRU()
  local encoder, opt = buildEncoder(onmt.BiEncoder, 'GRU')
  genericCheckSerial(encoder, opt)
end

function encoderTest.dbrnn_LSTM()
  local encoder, opt = buildEncoder(onmt.DBiEncoder, 'LSTM')
  genericCheckDim(encoder, opt)
end

function encoderTest.dbrnn_masking_LSTM()
  local encoder, _ = buildEncoder(onmt.DBiEncoder, 'LSTM')
  genericCheckMasking(encoder)
end

function encoderTest.dbrnn_saveAndLoad_LSTM()
  local encoder, opt = buildEncoder(onmt.DBiEncoder, 'LSTM')
  genericCheckSerial(encoder, opt)
end

function encoderTest.dbrnn_GRU()
  local encoder, opt = buildEncoder(onmt.DBiEncoder, 'GRU')
  genericCheckDim(encoder, opt)
end

function encoderTest.dbrnn_masking_GRU()
  local encoder, _ = buildEncoder(onmt.DBiEncoder, 'GRU')
  genericCheckMasking(encoder)
end

function encoderTest.dbrnn_saveAndLoad_GRU()
  local encoder, opt = buildEncoder(onmt.DBiEncoder, 'GRU')
  genericCheckSerial(encoder, opt)
end

function encoderTest.gnmt_LSTM()
  local encoder, opt = buildEncoder(onmt.GoogleEncoder, 'LSTM')
  genericCheckDim(encoder, opt)
end

function encoderTest.gnmt_masking_LSTM()
  local encoder, _ = buildEncoder(onmt.GoogleEncoder, 'LSTM')
  genericCheckMasking(encoder)
end

function encoderTest.gnmt_saveAndLoad_LSTM()
  local encoder, opt = buildEncoder(onmt.GoogleEncoder, 'LSTM')
  genericCheckSerial(encoder, opt)
end

function encoderTest.gnmt_GRU()
  local encoder, opt = buildEncoder(onmt.GoogleEncoder, 'GRU')
  genericCheckDim(encoder, opt)
end

function encoderTest.gnmt_masking_GRU()
  local encoder, _ = buildEncoder(onmt.GoogleEncoder, 'GRU')
  genericCheckMasking(encoder)
end

function encoderTest.gnmt_saveAndLoad_GRU()
  local encoder, opt = buildEncoder(onmt.GoogleEncoder, 'GRU')
  genericCheckSerial(encoder, opt)
end

function encoderTest.pdbrnn_LSTM()
  local encoder, opt = buildEncoder(onmt.PDBiEncoder, 'LSTM')
  genericCheckDim(encoder, opt)
end

function encoderTest.pdbrnn_saveAndLoad_LSTM()
  local encoder, opt = buildEncoder(onmt.PDBiEncoder, 'LSTM')
  genericCheckSerial(encoder, opt)
end

function encoderTest.pdbrnn_GRU()
  local encoder, opt = buildEncoder(onmt.PDBiEncoder, 'GRU')
  genericCheckDim(encoder, opt)
end

function encoderTest.pdbrnn_saveAndLoad_GRU()
  local encoder, opt = buildEncoder(onmt.PDBiEncoder, 'GRU')
  genericCheckSerial(encoder, opt)
end

function encoderTest.cnn()
  local encoder, opt = buildEncoder(onmt.CNNEncoder)
  genericCheckDim(encoder, opt)
end

function encoderTest.cnn_masking()
  local encoder, _ = buildEncoder(onmt.CNNEncoder)
  genericCheckMasking(encoder)
end

function encoderTest.cnn_saveAndLoad()
  local encoder, opt = buildEncoder(onmt.CNNEncoder)
  genericCheckSerial(encoder, opt)
end

return encoderTest
