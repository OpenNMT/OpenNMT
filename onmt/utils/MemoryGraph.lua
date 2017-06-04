--[[
  MemoryGraph is a class for finding and optimizing memory usage (storage) dependency within all the computation graphs
]]

local MemoryGraph = torch.class('MemoryGraph')

MemoryGraph.protectedBuffer = {
  -- We cannot share the output of these modules as they use it in their backward pass.
  output = {
  'nn.Sigmoid',
  'nn.SoftMax',
  'nn.Tanh'
  },
  -- We cannot share the input of these modules as they use it in their backward pass.
  input = {
  'nn.Linear',
  'nn.JoinTable',
  'nn.CMulTable',
  'nn.MM'
  }
}

function MemoryGraph.protected(m, buffer)
  local function contains(list, modName)
    if list then
      for i = 1, #list do
        if modName == list[i] then
          return true
        end
      end
    end
    return false
  end
  return contains(MemoryGraph.protectedBuffer[buffer], torch.typename(m))
end

-- inspired from nngraph.simple_print
local function removeNodeFromEdges(node_id, edges)
  local from_nodes = {}
  local to_nodes = {}
  -- remove edges
  local idx = 1
  while idx <= #edges do
   local edge = edges[idx]
   if edge.source == node_id then
    local to_node = edges[idx].target
    table.insert(to_nodes, to_node)
    table.remove(edges, idx)
   elseif edge.target == node_id then
    local from_node = edges[idx].source
    table.insert(from_nodes, from_node)
    table.remove(edges, idx)
   else
    idx = idx + 1
   end
  end

  -- add new edges
  for _, f in pairs(from_nodes) do
   for _, t in pairs(to_nodes) do
    local edge = {source = f, target= t}
    table.insert(edges, edge)
   end
  end

  return edges
end

local function isNodeGood(node)
  return node.data and node.data.module and torch.typename(node.data.module) ~= 'nn.Identity'
end

local function reIndexNodes(nodes, edges)
  -- make reverse map
  local rev_map = {}
  local map_from = {}
  local map_to = {}
  for idx = 1, #nodes do
   rev_map[nodes[idx].id] = idx
   nodes[idx].id = idx
   table.insert(map_from,0)
   table.insert(map_to,0)
  end
  for idx = 1, #edges do
   local edge = edges[idx]
   edge.source = rev_map[edge.source]
   edge.target = rev_map[edge.target]
   map_from[edge.source] = map_from[edge.source]+1
   map_to[edge.target] = map_to[edge.target]+1
  end
  local first = 0
  local last = 0
  -- find first and last nodes of the graph
  for idx = 1, #nodes do
   if map_from[idx] == 0 then
    assert(last==0)
    last=idx
   end
   if map_to[idx] == 0 then
    assert(first==0)
    first=idx
   end
  end
  return nodes, edges, first, last
end

local function cleanGraph(nodes, edges)
  local map_from = {}
  local map_to = {}
  for idx = 1, #nodes do
   map_from[nodes[idx].id]=0
   map_to[nodes[idx].id]=0
  end
  for idx = 1, #edges do
   local edge=edges[idx]
   map_from[edge.source] = map_from[edge.source]+1
   map_to[edge.target] = map_to[edge.target]+1
  end
  local idx2 = 1
  while idx2 <= #nodes do
   local node = nodes[idx2]
   if isNodeGood(node.orig_node) or map_from[node.id]==0 or map_to[node.id]==0 then
    idx2 = idx2 + 1
   else
    local id = node.id
    table.remove(nodes, idx2)
    edges = removeNodeFromEdges(id, edges)
   end
  end
  -- remove duplicated edges
  local all_edges = {}
  local new_edges = {}
  for idx = 1, #edges do
    local k = edges[idx].source .. '-' .. edges[idx].target
    if not all_edges[k] then
      all_edges[k] = 1
      table.insert(new_edges, edges[idx])
    end
  end

  return reIndexNodes(nodes, new_edges)
end

-- transform (recursively a nngraph into a nodes/edges graph)
local function loadGraph(graph)
  local nodes = {}
  local edges = {}
  local first
  local last

  for _, node in ipairs(graph.fg.nodes) do
   local idx = node.id
   local processed = false
   if node.data and node.data.module and
      (torch.isTypeOf(node.data.module, "onmt.Network") or torch.isTypeOf(node.data.module, "nn.gModule")) then
    local vnode = node.data.module.net or node.data.module
    if torch.type(vnode) == 'nn.gModule' then
     local vnodes, vedges, vfirst, vlast = loadGraph(vnode)
     for _, vn in ipairs(vnodes) do
      local vidx = idx .. '_' .. vn.id
      if vn.id == vfirst then vidx = idx end
      vn.orig_node.id = idx .. '_' .. vn.orig_node.id
      table.insert(nodes, {id = vidx, orig_node = vn.orig_node})
     end
     for _, ve in ipairs(vedges) do
      local vidx_source = idx .. '_' .. ve.source
      if ve.source == vfirst then vidx_source = idx end
      table.insert(edges, {source = vidx_source, target = idx .. '_' .. ve.target})
     end
     for ich = 1, #node.children do
      table.insert( edges, {source = idx .. '_' .. vlast, target = node.children[ich].id})
     end
     processed = true
    end
   end
   if not processed then
    table.insert(nodes, {id=idx, orig_node = node} )
    for ich = 1, #node.children do
      table.insert( edges, {source = idx, target = node.children[ich].id})
    end
   end
  end

  nodes, edges, first, last = cleanGraph(nodes, edges)
  return nodes , edges, first, last
end

function MemoryGraph:__init()
  self.graphs = {}
end

--[[
  Register a new module
]]
function MemoryGraph:add(name, gModule)
  if torch.type(gModule) == "nn.gModule" then
    local nodes , edges, first, last = loadGraph(gModule)

    -- index edges forward and backward
    local fromEdgeMap = {}
    local toEdgeMap = {}

    for _, e in ipairs(edges) do
      if not fromEdgeMap[e.source] then
        fromEdgeMap[e.source] = {}
      end
      table.insert(fromEdgeMap[e.source], e.target)
      if not toEdgeMap[e.target] then
        toEdgeMap[e.target] = {}
      end
      table.insert(toEdgeMap[e.target], e.source)
    end

    table.insert(self.graphs, { name=name, nodes=nodes, edges=edges, first=first, last=last,
                                fromEdgeMap=fromEdgeMap, toEdgeMap=toEdgeMap })
  end
end

-- first input and output modules of a graph (the first module nodes)
local function findInputModules(graph, nId, input_modules)
  local n = graph.nodes[nId]
  if n.orig_node.data and n.orig_node.data.module then
    input_modules[nId]=true
    return
  end
  for _, to in ipairs(graph.fromEdgeMap[nId]) do
    findInputModules(graph, to, input_modules)
  end
end

local function findOutputModules(graph, nId, output_modules)
  local n = graph.nodes[nId]
  if n.orig_node.data and n.orig_node.data.module then
    output_modules[nId]=true
    return
  end
  for _, from in ipairs(graph.toEdgeMap[nId]) do
    findOutputModules(graph, from, output_modules)
  end
end

function MemoryGraph:differentComputationGraph(st1, st2)
  local graphs1 = {}
  local direction1 = {}
  for _, i1 in ipairs(st1) do
    graphs1[i1[1]] = true
    local dir = i1[3] == "gradInput" and "bwd" or "fwd"
    direction1[dir] = true
  end
  local foundSameGraph = false
  local foundSameDirection = false
  local idx2 = 1
  while (not foundSameGraph or not foundSameDirection) and idx2 <= #st2 do
    foundSameGraph = graphs1[st2[idx2][1]]
    local dir = st2[idx2][3] == "gradInput" and "bwd" or "fwd"
    foundSameDirection = direction1[dir]
    idx2 = idx2 + 1
  end
  if not foundSameGraph or not foundSameDirection then
    return true
  end
  return false
end

-- check if we can reach n2-b2 from n1-b1 - forward and the minimal distance for doing so
function MemoryGraph:pathReachFwd(gid, n1, b1, n2, b2)
  if n1 == n2 then
    if b1 == "output" and b2 == "input" then return false end
    if b1 == "input" and b2 == "output" then return 1 end
    return 0
  end
  if self.graphs[gid].fromEdgeMap[n1] then
    local ld = 100
    for _, to in ipairs(self.graphs[gid].fromEdgeMap[n1]) do
      local d = self:pathReachFwd(gid, to, "output", n2, b2)
      if not d then return false end
      if d < ld then ld = d end
    end
    return ld + 1
  end
  return false
end

-- check if we can reach n2-b2 from n1-b1 - backward
function MemoryGraph:pathReachBwd(gid, n1, n2)
  if n1 == n2 then return 0 end
  if self.graphs[gid].toEdgeMap[n1] then
    local ld = 100
    for _, from in ipairs(self.graphs[gid].toEdgeMap[n1]) do
      local d = self:pathReachBwd(gid, from, n2)
      if not d then return false end
      if d < ld then ld = d end
    end
    return ld + 1
  end
  return false
end

-- st1 inherits from st2 if all instances of st1 reach st2, in same graphs and direction, in more than 2 operations
function MemoryGraph:inherits(st1, st2)
  for _, i1 in ipairs(st1) do
    for _, i2 in ipairs(st2) do
      -- same graph
      if i1[1] == i2[1] then
        if i1[3] == "gradInput" and i2[3] == "gradInput" then
          local d = self:pathReachBwd(i1[1], i1[2], i2[2])
          if not d or d < 2 then return false end
        elseif i1[3] ~= "gradInput" and i2[3] ~= "gradInput" then
          local d = self:pathReachFwd(i1[1], i1[2], i1[3], i2[2], i2[3])
          if not d or d < 2 then return false end
        end
      end
    end
  end
  return true
end

function MemoryGraph:optimize()
  -- for each storage find where it is used (graph, node - input/output/gradInput), and if it is protected
  local storageMap = {}
  local function register(t, protected, gid, nid, bid, norigid)
    if torch.isTensor(t) and t:storage() then
      local ptr = torch.pointer(t:storage())
      if not storageMap[ptr] then
        storageMap[ptr] = {protected=false, st=t:storage(),size=t:storage():size()*t:elementSize()}
      end
      storageMap[ptr].protected = storageMap[ptr].protected or protected
      table.insert(storageMap[ptr], {gid, nid, bid, norigid})
    elseif torch.type(t) == 'table' then
      for _, v in ipairs(t) do
        register(v, protected, gid, nid, bid, norigid)
      end
    end
  end
  -- main function checking if a storage can be shared
  local function shareableStorages(ptr1, ptr2)
    -- 2 storages can be shared if:
    -- a/ none of their instance are on the same graph, or same fwd/backward path
    -- b/ if one inherits completely from the other, with at least 2 operations in between
    if self:differentComputationGraph(storageMap[ptr2], storageMap[ptr1]) then
      return true
    end
    if self:inherits(storageMap[ptr1], storageMap[ptr2]) or self:inherits(storageMap[ptr2], storageMap[ptr1]) then
      return true
    end
    return false
  end

  for graphId, g in ipairs(self.graphs) do
    local input_modules = {}
    local output_modules = {}
    findInputModules(g, g.first, input_modules)
    findOutputModules(g, g.last, output_modules)
    for nodeId, n in ipairs(g.nodes) do
      if n.orig_node.data.module then
        for _, buf in ipairs({"input", "output", "gradInput"}) do
          -- some storage are protected because will be reuse for some modules in backward pass
          local protected = MemoryGraph.protected(n.orig_node.data.module, buf)
          -- to be safe, protect also input and output buffers of the complete graph
          protected = protected or (buf == "input" and input_modules[nodeId])
          protected = protected or (buf == "output" and output_modules[nodeId])
          register(n.orig_node.data[buf] or n.orig_node.data.module[buf], protected, graphId, nodeId, buf, n.orig_node.id)
        end
      end
    end
  end

  -- go through all unprotected storage and assign to each of them a cluster id
  local shareStorageMap = {}
  local storageCluster = {}
  local totalSize = 0
  local saveSize = 0
  local protectedSize = 0
  for ptr, storage in pairs(storageMap) do
    local storageType = torch.typename(storage.st) .. ':' .. storage.size
    totalSize = totalSize + storage.size
    if not storage.protected then
      if not shareStorageMap[storageType] then
        shareStorageMap[storageType] = {}
      end
      local i = 1
      local foundShareCluster
      while not foundShareCluster and i <= #shareStorageMap[storageType] do
        local clusterptr = shareStorageMap[storageType][i]
        local clusterisok = true
        local j = 1
        while clusterisok and j <= #storageCluster[clusterptr] do
          clusterisok=shareableStorages(storageCluster[clusterptr][j], ptr)
          j = j + 1
        end
        foundShareCluster = clusterisok and clusterptr
        i = i + 1
      end
      if foundShareCluster then
        saveSize = saveSize + storage.size
        table.insert(storageCluster[foundShareCluster], ptr)

      else
        table.insert(shareStorageMap[storageType], ptr)
        storageCluster[ptr] = { ptr }
      end
      storage.clusterId = #storageCluster
    else
      protectedSize = protectedSize + storage.size
    end
  end

  if _G.logger:_isVisible("DEBUG") then
    _G.logger:debug("MemoryGraph optimized clusters")
    for _, cl in pairs(storageCluster) do
      local cluster = { }
      for _, ptr in ipairs(cl) do
        local ts = {}
        for _, t in ipairs(storageMap[ptr]) do
          table.insert(ts,'('..table.concat({self.graphs[t[1]].name,t[4],t[3]}, ':')..')')
        end
        table.insert(cluster,table.concat(ts, ','))
      end
      _G.logger:debug(table.concat(cluster, ' '))
    end
  end

  -- assign to each tensor its share index
  local function share(t)
    if torch.isTensor(t) and t:storage() then
      local ptr = torch.pointer(t:storage())
      if storageMap[ptr].protected then
        return false
      end
      return storageMap[ptr].clusterId
    elseif torch.type(t) == 'table' then
      local shareIdx = {}
      for _, v in ipairs(t) do
        table.insert(shareIdx, share(v))
      end
      return shareIdx
    end
  end

  -- last, go back through all of the modules and assigned sharing ids to output and gradInput table or tensors
  for _, g in ipairs(self.graphs) do
    for _, n in ipairs(g.nodes) do
      if n.orig_node.data.module then
        local m = n.orig_node.data.module
        m.outputSharedIdx = share(m.output)
        m.gradInputSharedIdx = share(m.gradInput)
      end
    end
  end

  _G.logger:info('MemoryGraph optimization: total %d, protected %d (%0.2f%%), saved %d (%0.2f%%)',
                  totalSize, protectedSize, 100*protectedSize/totalSize, saveSize, 100*saveSize/totalSize)
  return totalSize, protectedSize, saveSize
end

--[[
  Dump the registered graphs as .dot files
]]
function MemoryGraph:dump(path)
  for _, g in ipairs(self.graphs) do
    local filename = path..'/'..g.name..'.dot'
    local file = io.open(filename, "w")
    assert(file, "cannot open '"..filename.."' for writing")

    local str = {}
    table.insert(str,'digraph G {\n')
    table.insert(str,'labelloc="t";\nlabel="' .. filename .. '";\n')
    table.insert(str,'node [shape = oval]; ')
    local nodelabels = {}
    for i, node in ipairs(g.nodes) do
      local true_node = node.orig_node
      local label = true_node:label()
      label = string.gsub(label, 'reverseMap = .*', '')
      local l =  '"' .. ( 'Node ' .. true_node.id .. '\\n' .. label ) .. '"'
      nodelabels[i] = 'n' .. true_node.id
      table.insert(str, '\n' .. nodelabels[i] .. '[label=' .. l .. '];')
    end
    table.insert(str,'\n')
    for _, edge in ipairs(g.edges) do
      table.insert(str,nodelabels[edge.source] .. ' -> ' .. nodelabels[edge.target] .. ';\n')
    end
    table.insert(str,'}')

    file:write(table.concat(str,''))
    file:close()
  end
end

return MemoryGraph
