local function clone_many_times(net, T)
  local clones = {}

  local params, gradParams
  if net.parameters then
    params, gradParams = net:parameters()
    if params == nil then
      params = {}
    end
  end

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
    end

    clones[t] = clone
    collectgarbage()
  end

  mem:close()
  return clones
end

local function reset_state(state, batch_l, t)
  local new_state, table_to_fill
  if t == nil then
    new_state = {}
    table_to_fill = new_state
  else
    new_state = {[t] = {}}
    table_to_fill = new_state[t]
  end

  for i = 1, #state do
    state[i]:zero()
    table.insert(table_to_fill, state[i][{{1, batch_l}}])
  end
  return new_state
end

local function copy_state(proto, state, batch_l)
  local st = reset_state(proto, batch_l)
  for i = 1, #proto do
    st[i]:copy(state[i])
  end
  return st
end

local function recursiveClone(out)
  if torch.isTensor(out) then
    return out:clone()
  else
    local res = {}
    for k, v in ipairs(out) do
      res[k] = recursiveClone(v)
    end
    return res
  end
end

return {
  clone_many_times = clone_many_times,
  reset_state = reset_state,
  copy_state = copy_state,
  recursiveClone = recursiveClone
}
