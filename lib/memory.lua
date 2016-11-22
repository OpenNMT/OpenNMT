local model_utils = require 'lib.utils.model_utils'

local Memory = {
  optnet = nil
}

local function init()
  local res, ret = pcall(function()
    return require 'optnet'
  end)
  if not res then
    print("WARNING: optnet module not installed - disabling optnet optimization: ")
  else
    Memory.optnet = ret
  end
  return Memory
end

function Memory.optimize(model, batch)
  if not Memory.optnet then return end
  -- record actual size of the batch
  local actual_batchsize = { source_length = batch.source_length, target_length = batch.target_length }
  -- batch of one single word since we optimize the first clone
  batch.source_length = 1
  batch.target_length = 1

  -- change forward function for all of the models to intercept actual inputs for optimize function
  local model_desc = {}
  for name, mod in pairs(model) do
    model_desc[name] = {}
    local net
    if mod.net then
      net = mod:net(1)
    else
      net = mod.network
    end
    model_desc[name]['net'] = net
    model_desc[name]['forward'] = net.forward
    net.forward = function(net, input)
      model_desc[name]['input'] = model_utils.recursiveClone(input)
      return model_desc[name]['forward'](net, input)
    end
  end

  -- initialize the network with a first batch - first time to get actual inputs from the networks
  local enc_states, context = model.encoder:forward(batch)
  local dec_outputs = model.decoder:forward(batch, enc_states, context)
  dec_outputs = model_utils.recursiveClone(dec_outputs)
  local enc_grad_states_out, grad_context, loss = model.decoder:backward(batch, dec_outputs, model.generator)
  model.encoder:backward(batch, enc_grad_states_out, grad_context)

  for name, mod in pairs(model_desc) do
    local net = mod.net
    print(name..' memory',Memory.optnet.countUsedMemory(net))
  end

  -- restore forward methods and optimize
  for name, mod in pairs(model_desc) do
    local net = mod.net
    net.forward = mod['forward']
    Memory.optnet.optimizeMemory(net, mod['input'], {mode='training'})
  end

  for name, mod in pairs(model_desc) do
    local net = mod.net
    print(name..' memory',Memory.optnet.countUsedMemory(net))
  end

  -- restore batch
  batch.source_length = actual_batchsize.source_length
  batch.target_length = actual_batchsize.target_length
end

return init()
