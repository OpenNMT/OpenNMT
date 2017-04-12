require('onmt.init')

local tester = ...

local encoderTest = torch.TestSuite()

local inputNet = nn.LookupTable(10, 20)
inputNet.inputSize = 20

local function buildEncoder(class)
  local cmd = onmt.utils.ExtendedCmdLine.new()
  class.declareOpts(cmd)

  local opt = cmd:parse('')
  opt.rnn_size = 30
  opt.dropout = 0

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
  tester:eq(#states, opt.layers * (opt.rnn_type == 'LSTM' and 2 or 1))
  for _, v in ipairs(states) do
    tester:eq(v:size(), torch.LongStorage({3, opt.rnn_size}))
  end

  local gradContextOutput = context:clone():uniform(-0.1, 0.1)
  local gradStatesOutput = {}
  for _, v in ipairs(states) do
    table.insert(gradStatesOutput, v:clone():uniform(-0.1, 0.1))
  end

  local gradInputs = encoder:backward(batch, gradStatesOutput, gradContextOutput)

  tester:eq(#gradInputs, 4)
  for _, v in ipairs(gradInputs) do
    tester:eq(v:size(), torch.LongStorage({3}))
  end

  return states, context
end

local function genericCheckMasking(encoder)
  local src = {
    torch.IntTensor({5, 6, 7, 8}),
    torch.IntTensor({5, 6}),
    torch.IntTensor({5, 6, 7}),
  }

  local batch = onmt.data.Batch.new(src)

  encoder:maskPadding()

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

function encoderTest.simple()
  local encoder, opt = buildEncoder(onmt.Encoder)
  genericCheckDim(encoder, opt)
end

function encoderTest.simple_masking()
  local encoder, _ = buildEncoder(onmt.Encoder)
  genericCheckMasking(encoder)
end

function encoderTest.simple_saveAndLoad()
  local encoder, opt = buildEncoder(onmt.Encoder)
  genericCheckSerial(encoder, opt)
end

function encoderTest.brnn()
  local encoder, opt = buildEncoder(onmt.BiEncoder)
  genericCheckDim(encoder, opt)
end

function encoderTest.brnn_masking()
  local encoder, _ = buildEncoder(onmt.BiEncoder)
  genericCheckMasking(encoder)
end

function encoderTest.brnn_saveAndLoad()
  local encoder, opt = buildEncoder(onmt.BiEncoder)
  genericCheckSerial(encoder, opt)
end

function encoderTest.dbrnn()
  local encoder, opt = buildEncoder(onmt.DBiEncoder)
  genericCheckDim(encoder, opt)
end

function encoderTest.dbrnn_masking()
  local encoder, _ = buildEncoder(onmt.DBiEncoder)
  genericCheckMasking(encoder)
end

function encoderTest.dbrnn_saveAndLoad()
  local encoder, opt = buildEncoder(onmt.DBiEncoder)
  genericCheckSerial(encoder, opt)
end

function encoderTest.pdbrnn()
  local encoder, opt = buildEncoder(onmt.PDBiEncoder)
  genericCheckDim(encoder, opt)
end

function encoderTest.pdbrnn_saveAndLoad()
  local encoder, opt = buildEncoder(onmt.PDBiEncoder)
  genericCheckSerial(encoder, opt)
end

return encoderTest
