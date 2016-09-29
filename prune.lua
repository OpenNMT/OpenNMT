require 'nn'
require 'string'
require 'nngraph'

require 's2sa.models'
require 's2sa.data'
require 's2sa.plinear'

local cmd = torch.CmdLine()
-- file location
cmd:option('-model', 'model.t7','model file')
cmd:option('-savefile', 'model-pruned.t7','pruned model file')
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-ratio', 0.2, [[ratio of the parameters to prune]])
cmd:option('-prune', 'blind', [[type of pruning strategy: (class) blind, uniform]])

local opt = cmd:parse(arg)
local allparams=torch.Tensor()

-- find n-th percentile of a tensor without sorting the tensor
local function getKth(k, t)
  -- define 1000 buckets to avoid complete sorting for interval i/1000;i+1/1000
  local buckets={}
  for _=1,1000 do
    table.insert(buckets,{})
  end
  -- fill the bucket with the idx
  for i=1,t:size(1) do
    local nbucket=math.floor(t[i]*1000)
    if nbucket>999 then nbucket=999 end
    table.insert(buckets[nbucket+1],i)
  end
  -- find the bucket of interest
  local idx=1
  while k>0 do
    k=k-#buckets[idx]
    idx=idx+1
  end
  -- sort the bucket
  k=k+#buckets[idx-1]
  local sbucket=torch.Tensor(#buckets[idx-1])
  for i=1,#buckets[idx-1] do
    sbucket[i]=t[buckets[idx-1][i]]
  end
  return torch.sort(sbucket)[k]
end

-- count total number of parameters
local function countParameters(m)
  local classname=torch.typename(m)
  if classname=='nn.Linear' or classname=='nn.LinearNoBias' then
    local p=m:getParameters()
    if allparams:dim()==0 then
      allparams=p
    else
      allparams=torch.cat(allparams, p)
    end
  end
end

local function prune(m, gthreshold, locopt)
  local p=m:getParameters()
  local classname=torch.typename(m)
  if classname=='nn.Linear' or classname=='nn.LinearNoBias' then
    local lthreshold=getKth(math.floor(locopt.ratio*p:size(1))+1,torch.abs(p))
    local threshold=gthreshold
    if locopt.prune == 'uniform' then
      threshold=lthreshold
    end
    local pruned,total=m:prune(threshold)
    print(m.name,pruned,total)
  end
end

local function main()
  if not(opt.prune=='blind' or opt.prune=='uniform') then
    print('ERROR - prune type should be "blind" or "uniform"')
    return
  end
  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
  end
  print('loading model ' .. opt.model)
  local checkpoint = torch.load(opt.model)
  local model, model_opt = checkpoint[1], checkpoint[2]

  if model_opt.prune ~= nil then
    print('cannot prune pruned model')
    return
  end

  -- count all parameters
  for i = 1, #model do
    if model[i].apply then
      model[i]:apply(countParameters)
    end
  end

  -- sort by magnitude
  local gthreshold=getKth(math.floor(opt.ratio*allparams:size(1))+1,torch.abs(allparams))
  print('#parameters:', allparams:size(1), 'global threshold:',gthreshold, 'prune method:',opt.prune)

  -- apply pruning
  for i = 1, #model do
    model[i]:apply(function(m) prune(m, gthreshold, opt) end)
  end

  if model_opt.pruning == nil then
    model_opt.pruning = {}
  end

  table.insert(model_opt.pruning, opt)

  print('saving model to ' .. opt.savefile)
  torch.save(opt.savefile, {model, model_opt})

end

main()

