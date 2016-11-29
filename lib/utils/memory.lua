local Memory = {}

local supportedModules = {
  'nn.Linear',
  'nn.CMulTable',
  'nn.MM',
  'nn.Sum'
}

local function _isSupported(m)
  for i = 1, #supportedModules do
    if torch.typename(m) == supportedModules[i] then
      return true
    end
  end
  return false
end

local function _tensorIncluded(t, l)
  if torch.isTensor(l) then
    return torch.pointer(t:storage()) == torch.pointer(l:storage())
  elseif torch.type(l) == 'table' then
    for _, m in ipairs(l) do
      if _tensorIncluded(t, m) then return true end
    end
  end
  return false
end

-- we cannot share a tensor if it is exposed/coming from outside of the net otherwise we could generate side-effects
local function _canShare(t, net, protected)
  if torch.isTensor(t) and t:storage() then
    if not _tensorIncluded(t, net.gradInput) and not _tensorIncluded(t, net.output) and not _tensorIncluded(t, protected) then
      return true
    end
  elseif torch.type(t) == 'table' then
    for _, m in ipairs(t) do
      if not _canShare(m, net, protected) then
        return false
      end
    end
    return true
  end
  return false
end

local function _size(t, mempool)
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
      size = size + _size(m, mempool)
    end
  end
  return size
end

function Memory.optimize(model, criterion, batch, verbose)
  if verbose then
    print('Preparing memory optimization...')
  end

  -- record actual size of the batch
  local actual_batchsize = { source_length = batch.source_length, target_length = batch.target_length }

  -- batch of one single word since we optimize the first clone
  batch.source_length = 1
  batch.target_length = 1

  local model_desc = {}

  local function registerNet(store, net)
    store['net'] = net
    store['forward'] = net.forward
    net.forward = function(net, input)
      store['input'] = input
      return store['forward'](net, input)
    end
    store['backward'] = net.backward
    net.backward = function(net, input, gradOutput)
      store['gradOutput'] = gradOutput
      return store['backward'](net, input, gradOutput)
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

  -- initialize the network with a first batch
  local enc_states, context = model.encoder:forward(batch)
  local dec_outputs = model.decoder:forward(batch, enc_states, context)
  dec_outputs = utils.Tensor.recursiveClone(dec_outputs)
  local enc_grad_states_out, grad_context, loss = model.decoder:backward(batch, dec_outputs, criterion)
  model.encoder:backward(batch, enc_grad_states_out, grad_context)

  local totSize = 0
  local sharedSize = 0
  local idx = 1
  for name, desc in pairs(model_desc) do
    for i = 1, #desc do
      local net = desc[i]['net']
      local mempool = {}

      -- some modules are using output when performing updateGradInput - so we cannot share these
      local protectedOutput = { desc[i]['input'] }
      net:apply(function(m)
          if m.output and not _isSupported(m) then
            table.insert(protectedOutput, m.output)
        end
      end)

      net:apply(function(m)
        local giSize = _size(m.gradInput, mempool)
        local oSize = _size(m.output, mempool)
        totSize = totSize + giSize
        totSize = totSize + oSize
        if _canShare(m.gradInput, net, desc[i]['gradOutput']) then
          sharedSize = sharedSize + giSize
          m.gradInputSharedIdx = idx
          idx = idx + 1
        end
        if _canShare(m.output, net, protectedOutput) then
          sharedSize = sharedSize + oSize
          m.outputSharedIdx = idx
          idx = idx + 1
        end
      end)

      -- restore function on network backward/forward interception input
      net.backward = nil
      net.forward = nil
    end
  end

  if verbose then
    print(string.format(' * sharing %d%% of output/gradInput tensors memory between clones', (sharedSize / totSize)*100))
  end

  -- restore batch
  batch.source_length = actual_batchsize.source_length
  batch.target_length = actual_batchsize.target_length
end

return Memory
