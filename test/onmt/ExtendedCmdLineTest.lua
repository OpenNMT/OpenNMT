require('onmt.init')

local tester = ...

local cmdLineTest = torch.TestSuite()

function cmdLineTest.convert_simple()
  local cmd = onmt.utils.ExtendedCmdLine.new()

  tester:eq(cmd:convert('key', '1'), '1')
  tester:eq(cmd:convert('key', '1', 'string'), '1')
  tester:eq(cmd:convert('key', '1', 'number'), 1)
end

function cmdLineTest.convert_boolean()
  local cmd = onmt.utils.ExtendedCmdLine.new()

  tester:eq(cmd:convert('key', '1', 'boolean'), true)
  tester:eq(cmd:convert('key', '0', 'boolean'), false)
  tester:eq(cmd:convert('key', 'true', 'boolean'), true)
  tester:eq(cmd:convert('key', 'false', 'boolean'), false)
end

function cmdLineTest.convert_table()
  local cmd = onmt.utils.ExtendedCmdLine.new()

  tester:eq(cmd:convert('key', '1', 'table', 'number'), { 1 })
  tester:eq(cmd:convert('key', '1', 'table', 'string'), { '1' })
  tester:eq(cmd:convert('key', '1 2', 'table', 'number'), { 1, 2 })
  tester:eq(cmd:convert('key', '1,2', 'table', 'number'), { 1, 2 })
  tester:eq(cmd:convert('key', '1,2', 'string'), '1,2')
end

function cmdLineTest.dumpAndLoadOptions()
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.Seq2Seq.declareOpts(cmd)

  local opt = cmd:parse('')
  opt.src_word_vec_size = { 500, 200 }

  cmd:dumpConfig(opt, 'config.txt')
  opt._is_default = nil

  local cmd2 = onmt.utils.ExtendedCmdLine.new()
  onmt.Seq2Seq.declareOpts(cmd2)

  local opt2 = cmd2:parse('')

  cmd2:loadConfig('config.txt', opt2)
  opt2._is_default = nil

  tester:eq(opt2, opt)
  os.remove('config.txt')
end

function cmdLineTest.parse_simple()
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.Seq2Seq.declareOpts(cmd)

  local opt = cmd:parse({'-rnn_size', '200', '-layers', '1'})

  tester:eq(opt.rnn_size, 200)
  tester:eq(opt.layers, 1)
end

function cmdLineTest.parse_boolean()
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.Seq2Seq.declareOpts(cmd)

  local opt = cmd:parse({'-brnn'})
  tester:eq(opt.brnn, true)

  opt = cmd:parse({'-brnn', '1'})
  tester:eq(opt.brnn, true)

  opt = cmd:parse({'-brnn', 'true'})
  tester:eq(opt.brnn, true)

  opt = cmd:parse({'-brnn', '0'})
  tester:eq(opt.brnn, false)

  opt = cmd:parse({'-brnn', 'false'})
  tester:eq(opt.brnn, false)
end

function cmdLineTest.parse_multiValue()
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.Seq2Seq.declareOpts(cmd)

  local opt = cmd:parse({'-src_word_vec_size', '500', '200'})
  tester:eq(opt.src_word_vec_size, { 500, 200 })

  opt = cmd:parse({'-src_word_vec_size', '500,200'})
  tester:eq(opt.src_word_vec_size, { 500, 200 })

  opt = cmd:parse({'-src_word_vec_size', '500'})
  tester:eq(opt.src_word_vec_size, { 500 })
end

function cmdLineTest.fail_unknown()
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.Seq2Seq.declareOpts(cmd)

  tester:assertError(function() cmd:parse({'-src_word_vec_size', '500', '-xxx'}) end)
end

return cmdLineTest
