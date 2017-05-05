require('onmt.init')

local tester = ...

local optimTest = torch.TestSuite()

local function getOptim(args)
  local cmd = onmt.utils.ExtendedCmdLine.new()
  onmt.train.Optim.declareOpts(cmd)

  local opt = cmd:parse('')

  for k, v in pairs(args) do
    opt[k] = v
  end

  return onmt.train.Optim.new(opt)
end

function optimTest.decay_default_noDecay()
  local args = {
    decay = 'default',
    learning_rate = 1,
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(5.0, 2), args.learning_rate)
end

function optimTest.decay_default_decayByEpoch()
  local args = {
    decay = 'default',
    learning_rate = 1,
    start_decay_at = 2
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(5.0, 2), args.learning_rate * optim.args.learning_rate_decay)
end

function optimTest.decay_default_decayByPPL()
  local args = {
    decay = 'default',
    learning_rate = 1
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(11.0, 2), args.learning_rate * optim.args.learning_rate_decay)
end

function optimTest.decay_default_decayByPPLDelta()
  local args = {
    decay = 'default',
    learning_rate = 1,
    start_decay_ppl_delta = 0.5
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(10.75, 2), args.learning_rate * optim.args.learning_rate_decay)
end

function optimTest.decay_perplexityOnly_noDecay()
  local args = {
    decay = 'perplexity_only',
    learning_rate = 1,
    start_decay_at = 2
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(5.0, 2), args.learning_rate)
end

function optimTest.decay_perplexityOnly_decay()
  local args = {
    decay = 'perplexity_only',
    learning_rate = 1,
    start_decay_ppl_delta = 0.5
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(10.75, 2), args.learning_rate * optim.args.learning_rate_decay)
end

function optimTest.decay_epochOnly_noDecay()
  local args = {
    decay = 'epoch_only',
    learning_rate = 1,
    start_decay_at = 3
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(11.0, 2), args.learning_rate)
end

function optimTest.decay_epochOnly_decay()
  local args = {
    decay = 'epoch_only',
    learning_rate = 1,
    start_decay_at = 2
  }

  local optim = getOptim(args)

  optim.valPerf[1] = 10.0
  tester:eq(optim:updateLearningRate(9.0, 2), args.learning_rate * optim.args.learning_rate_decay)
end

-- Test custom optimization methods against torch.optim.
local ret, optim = pcall(require, 'optim')

if ret then

  local function validateOptimMethod(refFunc, customFunc, learningRate)
    local params = torch.FloatTensor(30):uniform()
    local gradParams = torch.FloatTensor(30):uniform()

    local config = {}
    if learningRate then
      config.learningRate = learningRate
    end

    local opfunc = function(_)
      return 5.0, gradParams:clone()
    end

    local refState = {}
    local refParams = params:clone()
    refParams = refFunc(opfunc, refParams, config, refState)

    local customState = {}
    local customParams = params:clone()
    local customGradParams = gradParams:clone()
    if learningRate then
      customFunc(customGradParams, learningRate, customState)
    else
      customFunc(customGradParams, customState)
    end
    customParams:add(customGradParams)

    tester:eq(customParams, refParams, 1e-6)

    -- Second round with initialized states.
    refParams = refFunc(opfunc, refParams, config, refState)

    customGradParams:copy(gradParams)
    if learningRate then
      customFunc(customGradParams, learningRate, customState)
    else
      customFunc(customGradParams, customState)
    end
    customParams:add(customGradParams)
    tester:eq(customParams, refParams, 1e-6)
  end

  function optimTest.adam()
    validateOptimMethod(optim.adam, onmt.train.Optim.adamStep, 0.1)
  end

  function optimTest.adagrad()
    validateOptimMethod(optim.adagrad, onmt.train.Optim.adagradStep, 0.1)
  end

  function optimTest.adadelta()
    validateOptimMethod(optim.adadelta, onmt.train.Optim.adadeltaStep)
  end

end

return optimTest
