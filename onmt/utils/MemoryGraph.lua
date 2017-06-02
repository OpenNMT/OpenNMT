--[[MemoryGraph is a class for finding memory usage dependency within a graph
]]
local MemoryGraph = torch.class('MemoryGraph')

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

local pidx=1
function plot(nodes, edges)
    local str = {}
  table.insert(str,'digraph G {\n')
  if title then
    table.insert(str,'labelloc="t";\nlabel="' .. title .. '";\n')
  end
  table.insert(str,'node [shape = oval]; ')
  local nodelabels = {}
  for i,node in ipairs(nodes) do
    local true_node = node.orig_node
    local l =  '"' .. ( 'Node' .. true_node.id .. '\\n' .. true_node:label() ) .. '"'
    nodelabels[node.id] = 'n' .. true_node.id
    table.insert(str, '\n' .. nodelabels[node.id] .. '[label=' .. l .. '];')
  end
  table.insert(str,'\n')
  for i,edge in ipairs(edges) do
    table.insert(str,nodelabels[edge.source] .. ' -> ' .. nodelabels[edge.target] .. ';\n')
  end
  table.insert(str,'}')
  local fgv = io.open('p'..pidx..'.dot','w')
  pidx=pidx+1
  fgv:write(table.concat(str,''))
  fgv:close()
end

local function loadGraph(graph)
  local nodes = {}
  local edges = {}
  for _, node in ipairs(graph.nodes) do
   local idx = node.id
   local processed = false
   if node.data and node.data.module and
      (torch.isTypeOf(node.data.module, "onmt.Network") or torch.isTypeOf(node.data.module, "nn.gModule")) then
    local vnode = node.data.module.net or node.data.module
    if torch.type(vnode) == 'nn.gModule' then
     local vnodes, vedges, vfirst, vlast = loadGraph(vnode.fg)
     for _, vn in ipairs(vnodes) do
      local vidx = idx .. '_' .. vn.id
      if vn.id == vfirst then vidx = idx end
      vn.orig_node.id = vidx
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

function MemoryGraph:__init(gModule, storageDep)
  self.nodes, self.edges, self.first, self.last = loadGraph(gModule)
end

function MemoryGraph:dump(filename)
  print('-->',filename)
  local file = io.open(filename, "w")
  assert(file, "cannot open '"..filename.."' for writing")

  local str = {}
  table.insert(str,'digraph G {\n')
  if title then
    table.insert(str,'labelloc="t";\nlabel="' .. filename .. '";\n')
  end
  table.insert(str,'node [shape = oval]; ')
  local nodelabels = {}
  for i,node in ipairs(self.nodes) do
    local true_node = node.orig_node
    local l =  '"' .. ( 'Node' .. true_node.id .. '\\n' .. true_node:label() ) .. '"'
    nodelabels[i] = 'n' .. true_node.id
    table.insert(str, '\n' .. nodelabels[i] .. '[label=' .. l .. '];')
  end
  table.insert(str,'\n')
  for i,edge in ipairs(self.edges) do
    table.insert(str,nodelabels[edge.source] .. ' -> ' .. nodelabels[edge.target] .. ';\n')
  end
  table.insert(str,'}')

  file:write(table.concat(str,''))
  file:close()
end

return MemoryGraph
