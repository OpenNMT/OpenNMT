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

  local opt = cmd:parse({'-input_feed'})
  tester:eq(opt.input_feed, false)

  opt = cmd:parse({'-input_feed', '1'})
  tester:eq(opt.input_feed, true)

  opt = cmd:parse({'-input_feed', 'true'})
  tester:eq(opt.input_feed, true)

  opt = cmd:parse({'-input_feed', '0'})
  tester:eq(opt.input_feed, false)

  opt = cmd:parse({'-input_feed', 'false'})
  tester:eq(opt.input_feed, false)

  tester:eq(opt.fix_word_vecs_enc, false)
  opt = cmd:parse({'-fix_word_vecs_enc'})
  tester:eq(opt.fix_word_vecs_enc, true)
  opt = cmd:parse({'-fix_word_vecs_enc', 'true'})
  tester:eq(opt.fix_word_vecs_enc, true)
  opt = cmd:parse({'-fix_word_vecs_enc', '1'})
  tester:eq(opt.fix_word_vecs_enc, true)
  opt = cmd:parse({'-fix_word_vecs_enc', 'false'})
  tester:eq(opt.fix_word_vecs_enc, false)
  opt = cmd:parse({'-fix_word_vecs_enc', '0'})
  tester:eq(opt.fix_word_vecs_enc, false)
  opt = cmd:parse({'-fix_word_vecs_enc', 'pretrained'})
  tester:eq(opt.fix_word_vecs_enc, 'pretrained')
  tester:assertError(function() cmd:parse({'-fix_word_vecs_enc', 'xxx'}) end)
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
  onmt.data.SampledDataset.declareOpts(cmd)
  tester:assertError(function() cmd:parse({'-src_word_vec_size', '500', '-xxx'}) end)
  tester:assertError(function() cmd:parse({'-sample_tgt_vocab', '-sample', '0'}) end)
end

function cmdLineTest.arguments()
  local cmd = onmt.utils.ExtendedCmdLine.new()
  local options = {
    {
      'mode', 'string',
      [['score' apply lm to input text, 'sample' samples output based on input text.]],
      {
        enum = { 'score', 'sample' }
      }
    }
  }

  cmd:setCmdLineOptions(options, 'Data')
  tester:assertError(function() cmd:parse({''}) end)
  tester:assertNoError(function() cmd:parse({'score'}) end)
end

return cmdLineTest
