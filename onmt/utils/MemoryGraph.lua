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
  local idx = 1
  while idx <= #nodes do
   local node = nodes[idx]
   if isNodeGood(node.orig_node) or map_from[node.id]==0 or map_to[node.id]==0 then
    idx = idx + 1
   else
    local id = node.id
    table.remove(nodes, idx)
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
  local nodes , edges, first, last = loadGraph(gModule)
  table.insert(self.graphs, { name=name, nodes=nodes, edges=edges, first=first, last=last })
end

-- first input and output modules of a graph (the first module nodes)
local function findInputModules(graph, nId, input_modules)
  local n = graph.nodes[nId]
  if n.orig_node.data and n.orig_node.data.module then
    input_modules[nId]=true
    return
  end
  for _, e in ipairs(graph.edges) do
    if e.source == nId then
      findInputModules(graph, e.target, input_modules)
    end
  end
end

local function findOutputModules(graph, nId, output_modules)
  local n = graph.nodes[nId]
  if n.orig_node.data and n.orig_node.data.module then
    output_modules[nId]=true
    return
  end
  for _, e in ipairs(graph.edges) do
    if e.target == nId then
      findOutputModules(graph, e.source, output_modules)
    end
  end
end

function MemoryGraph:optimize()
  -- for each storage find where it is used (graph, node - input/output/gradInput), and if it is protected
  local storageMap = {}
  function register(t, protected, gid, nid, bid)
    if torch.isTensor(t) and t:storage() then
      local ptr = torch.pointer(t:storage())
      if not storageMap[ptr] then
        storageMap[ptr] = {protected=false, st=t:storage()}
      end
      storageMap[ptr].protected = storageMap[ptr].protected or protected
      table.insert(storageMap[ptr], {gid, nid, bid})
    elseif torch.type(t) == 'table' then
      for _, v in ipairs(t) do
        register(v, protected, gid, nid, bid)
      end
    end
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
          register(n.orig_node.data.module[buf], protected, graphId, nodeId, buf)
        end
      end
    end
  end
  -- go through all unprotected storage and share the ones with similar shape
  local shareStorageMap = {}
  local protectedStorageMap = {}
  local totalSize = 0
  local saveSize = 0
  for _, storage in pairs(storageMap) do
    local storageType = torch.typename(storage.st) .. ':' .. storage.st:size()
    totalSize = totalSize + storage.st:size()
    if not storage.protected then
      if not shareStorageMap[storageType] then
        shareStorageMap[storageType] = 0
      else
        saveSize = saveSize + storage.st:size()
      end
      shareStorageMap[storageType] = shareStorageMap[storageType] + 1
    else
      if not protectedStorageMap[storageType] then
        protectedStorageMap[storageType] = 0
      end
      protectedStorageMap[storageType] = protectedStorageMap[storageType] + 1
    end
  end
  for shape, count in pairs(shareStorageMap) do
    print('share',count,shape)
  end
  for shape, count in pairs(protectedStorageMap) do
    print('protect',count,shape)
  end
  print('total:', totalSize, 'save:', saveSize, 'ratio:', saveSize/totalSize)
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
    if title then
      table.insert(str,'labelloc="t";\nlabel="' .. filename .. '";\n')
    end
    table.insert(str,'node [shape = oval]; ')
    local nodelabels = {}
    for i,node in ipairs(g.nodes) do
      local true_node = node.orig_node
      local label = true_node:label()
      label = string.gsub(label, 'reverseMap = .*', '')
      local l =  '"' .. ( 'Node ' .. true_node.id .. '\\n' .. label ) .. '"'
      nodelabels[i] = 'n' .. true_node.id
      table.insert(str, '\n' .. nodelabels[i] .. '[label=' .. l .. '];')
    end
    table.insert(str,'\n')
    for i,edge in ipairs(g.edges) do
      table.insert(str,nodelabels[edge.source] .. ' -> ' .. nodelabels[edge.target] .. ';\n')
    end
    table.insert(str,'}')

    file:write(table.concat(str,''))
    file:close()
  end
end

return MemoryGraph
