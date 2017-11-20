require('onmt.init')

local tester = ...

local hookManagerTest = torch.TestSuite()

function hookManagerTest.nohook()
  local hookManager = onmt.utils.HookManager.new({})
  tester:eq(hookManager.hooks, {})
  hookManager = onmt.utils.HookManager.new({hook_file=''})
  tester:eq(hookManager.hooks, {})
end

function hookManagerTest.badhook()
  local logger_save = _G.logger
  _G.logger=nil
  local _, err = pcall(
    function()
      onmt.utils.HookManager.new({hook_file='bad'})
    end)
  tester:assert(err~=nil)
  _G.logger = logger_save
end

function hookManagerTest.options()
  local logger_save = _G.logger
  _G.logger=nil
  local hookManager
  local _, err = pcall(
    function()
      hookManager = onmt.utils.HookManager.new({hook_file='test.data.testhooks'})
      tester:ne(hookManager.hooks["declareOpts"], nil)
    end)
  tester:assert(err==nil)

  if hookManager then
    local cmd = onmt.utils.ExtendedCmdLine.new('train.lua')
    onmt.data.SampledVocabDataset.declareOpts(cmd)
    cmd:text('Other options')
    cmd:text('')
    onmt.utils.Memory.declareOpts(cmd)
    onmt.utils.Profiler.declareOpts(cmd)

    tester:assert(cmd.options['-sample_vocab']~=nil)
    tester:assert(cmd.options['-disable_mem_optimization']~=nil)
    tester:assert(cmd.options['-profiler']~=nil)
    tester:eq(cmd.options['-sample_vocab_type'], nil)
    tester:eq(cmd.options['-sample_vocab'].default, false)

    -- insert on the fly the option depending if there is a hook selected
    onmt.utils.HookManager.updateOpt({'-hook_file','test.data.testhooks'}, cmd)

    -- removed profiler option
    tester:eq(cmd.options['-profiler'], nil)
    -- new sample_vocab_type option
    tester:ne(cmd.options['-sample_vocab_type'], nil)
    -- sample_vocab true by default
    tester:eq(cmd.options['-sample_vocab'].default, true)
    -- new happy options
    tester:ne(cmd.options['-happy'], nil)
  end
  _G.logger = logger_save
end

function hookManagerTest.function_call()
  local logger_save = _G.logger
  local hookmanager_save = _G.hookManager
  _G.logger=nil
  local hookManager
  local _, err = pcall(
    function()
      hookManager = onmt.utils.HookManager.new({hook_file='test.data.testhooks'})
      tester:ne(hookManager.hooks["declareOpts"], nil)
    end)
  tester:assert(err==nil)

  if hookManager then
    local tokenizer = require('tools.utils.tokenizer')
    tester:ne(tokenizer.tokenize({segment_alphabet={}},"it is a test"), "XX")
    _G.hookManager = hookManager
    tester:eq(tokenizer.tokenize({segment_alphabet={}},"it is a test"), "XX")
  end
  _G.logger = logger_save
  _G.hookManager = hookmanager_save
end

return hookManagerTest
