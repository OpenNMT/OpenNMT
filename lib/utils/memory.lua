-- Allow tensor sharing for these modules.
local supportedModules = {
  'nn.Linear',
  'nn.CMulTable',
  'nn.MM',
  'nn.Sum'
}

local function isSupported(m)
  for i = 1, #supportedModules do
    if torch.typename(m) == supportedModules[i] then
      return true
    end
  end
  return false
end

local function tensorIncluded(t, l)
  if torch.isTensor(l) then
    return torch.pointer(t:storage()) == torch.pointer(l:storage())
  elseif torch.type(l) == 'table' then
    for _, m in ipairs(l) do
      if tensorIncluded(t, m) then
        return true
      end
    end
  end
  return false
end

-- We cannot share a tensor if it is exposed or coming from outside of the net
-- otherwise we could generate side-effects.
local function canShare(t, net, protected)
  if torch.isTensor(t) and t:storage() then
    if not tensorIncluded(t, net.gradInput) and not tensorIncluded(t, net.output) and not tensorIncluded(t, protected) then
      return true
    end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      if not canShare(m, net, protected) then
        return false
      end
    end
    return true
  end
  return false
end

local function getSize(t, mempool)
  local size=0
  if torch.isTensor(t) then
    if t:storage() then
      if not mempool[torch.pointer(t:storage())] then
        mempool[torch.pointer(t:storage())] = t:storage():size()*t:elementSize()
        return mempool[torch.pointer(t:storage())]
      end
    end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      size = size + getSize(m, mempool)
    end
  end
  return size
end

local Memory = {}

function Memory.optimize(model, criterion, batch, verbose)
  if verbose then
    print('Preparing memory optimization...')
  end

  -- Batch of one single word since we optimize the first clone.
  local realSizes = { source_length = batch.source_length, target_length = batch.target_length }

  batch.source_length = 1
  batch.target_length = 1

  local model_desc = {}

  -- Convenience function to register a network to optimize.
  local function registerNet(store, net)
    store['net'] = net
    store['forward'] = net.forward
    net.forward = function(network, input)
      store['input'] = input
      return store['forward'](network, input)
    end
    store['backward'] = net.backward
    net.backward = function(network, input, gradOutput)
      store['gradOutput'] = gradOutput
      return store['backward'](network, input, gradOutput)
    end
  end

  for name, mod in pairs(model) do
    model_desc[name] = {}

    if mod.net then
      -- If the module directly contains a network, take the first clone.
      model_desc[name][1] = {}
      registerNet(model_desc[name][1], mod:net(1))
    elseif mod.modules then
      -- Otherwise, look in submodules instead.
      for i = 1, #mod.modules do
        if mod.modules[i].net then
          model_desc[name][i] = {}
          registerNet(model_desc[name][i], mod.modules[i]:net(1))
        end
      end
    end
  end

  -- Initialize all intermediate tensors with a first batch.
  local enc_states, context = model.encoder:forward(batch)
  local dec_outputs = model.decoder:forward(batch, enc_states, context)
  dec_outputs = utils.Tensor.recursiveClone(dec_outputs)
  local enc_grad_states_out, grad_context, _ = model.decoder:backward(batch, dec_outputs, criterion)
  model.encoder:backward(batch, enc_grad_states_out, grad_context)

  local totSize = 0
  local sharedSize = 0
  for _, desc in pairs(model_desc) do
    for i = 1, #desc do
      local net = desc[i]['net']
      local mempool = {}

      -- Some modules are using output when performing updateGradInput so we cannot share these.
      local protectedOutput = { desc[i]['input'] }
      net:apply(function(m)
        if m.output and not isSupported(m) then
          table.insert(protectedOutput, m.output)
        end
      end)

      local idx = 1

      net:apply(function(m)
        local giSize = getSize(m.gradInput, mempool)
        local oSize = getSize(m.output, mempool)
        totSize = totSize + giSize
        totSize = totSize + oSize
        if canShare(m.gradInput, net, desc[i]['gradOutput']) then
          sharedSize = sharedSize + giSize
          m.gradInputSharedIdx = idx
          idx = idx + 1
        end
        if canShare(m.output, net, protectedOutput) then
          sharedSize = sharedSize + oSize
          m.outputSharedIdx = idx
          idx = idx + 1
        end
      end)

      -- Restore function on network backward/forward interception input.
      net.backward = nil
      net.forward = nil
    end
  end

  if verbose then
    print(string.format(' * sharing %d%% of output/gradInput tensors memory between clones', (sharedSize / totSize)*100))
  end

  -- Restore batch to be transparent for the calling code.
  batch.source_length = realSizes.source_length
  batch.target_length = realSizes.target_length
end

return Memory
