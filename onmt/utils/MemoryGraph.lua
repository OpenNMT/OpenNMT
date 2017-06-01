--[[MemoryGraph is a class for finding memory usage dependency within a graph
]]
local MemoryGraph = torch.class('MemoryGraph')

-- explore gModule to build unique global graph by exploring iteratively on gNodes
-- add mg_nodes and mg_parent to gModules
local function exploreGModule(currentGN, fromN, fromGN, nodes, gNodes, nodeMap)
  local function exploreNodes(current, from, currentgNode)
    local last
    local ptr = torch.pointer(current)
    local nodeId = nodeMap[ptr]
    if not nodeId then
      local m = current.data.module
      local last_node = from
      if m or #current.children==0 then
        table.insert(nodes, { ptr=ptr, from={}, gnode=currentgNode })
        table.insert(gNodes[currentgNode].mg_nodes, #nodes)
        nodeId = #nodes
        nodeMap[ptr] = nodeId
        nodes[nodeId].moduleName= m and torch.type(m) or '<>'
        last_node = #nodes
        if m and m.net and torch.type(m.net) == 'nn.gModule' then
          last_node = exploreGModule(m.net.fg, nodeId, currentgNode, nodes, gNodes, nodeMap)
        elseif torch.type(m) == 'nn.gModule' then
          last_node = exploreGModule(m.fg, nodeId, currentgNode, nodes, gNodes, nodeMap)
        end
      else
        nodeId = '<0>'
        nodes[nodeId] = { }
      end
      if current.children and #current.children>0 then
        for _, n in ipairs(current.children) do
          last = exploreNodes(n, last_node, currentgNode)
        end
      else
        last = nodeId
      end
      nodes[nodeId].last = last
    end
    if nodeId ~= '<0>' then
      table.insert(nodes[nodeId].from, from)
    end
    return nodes[nodeId].last
  end

  table.insert(gNodes, currentGN)
  currentGN.mg_nodes = {}
  currentGN.mg_parent = fromGN
  local roots = currentGN:roots()
  local last
  for _, r in ipairs(roots) do
    last = exploreNodes(r, fromN, #gNodes)
  end
  return last
end

function MemoryGraph:__init(gModule)
  self.nodes = {}
  self.gNodes = {}
  self.nodeMap = {}
  self.lastNode = exploreGModule(gModule, 'START', 0, self.nodes, self.gNodes, self.nodeMap)
end

function MemoryGraph:dump(filename)
  file = io.open(filename, "w")
  assert(file, "cannot open '"..filename.."' for writing")
  local gnStack = { 0 }
  file:write("digraph G { FINAL ; \n")
  for i,n in ipairs(self.gNodes) do
    while n.mg_parent ~= gnStack[#gnStack] do
      file:write('}\n')
      table.remove(gnStack)
    end
    file:write("subgraph cluster_"..i.." { color=blue; \n")
    for _, k in ipairs(n.mg_nodes) do
      file:write('  '..k..' [ label="'..self.nodes[k].moduleName..'" ];\n')
    end
    table.insert(gnStack, i)
  end
  while #gnStack > 1 do
    file:write('}\n')
    table.remove(gnStack)
  end
  for i,n in ipairs(self.nodes) do
    for _,j in ipairs(n.from) do
      file:write(j..' -> '..i..'\n')
    end
  end
  file:write(self.lastNode..' -> FINAL\n')
  file:write("}")
  file:close()
end
