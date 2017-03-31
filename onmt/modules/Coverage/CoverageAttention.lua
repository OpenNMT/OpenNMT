require('nngraph')

--[[ Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


    H_1 H_2 H_3 ... H_n
     q   q   q       q
      |  |   |       |
       \ |   |      /
           .....
         \   |  /
             a

Constructs a unit mapping:
  $$(H_1 .. H_n, q) => (a)$$
  Where H is of `batch x n x dim` and q is of `batch x dim`.

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

--]]
local CoverageAttention, parent = torch.class('onmt.CoverageAttention', 'onmt.Network')

--[[A nn-style module computing attention. Combining context coverage with gated context attention

  Parameters:

  * `dim` - dimension of the context vectors.
--]]
function CoverageAttention:__init(dim, coverageDim)
  parent.__init(self, self:_buildModel(dim, coverageDim))
end

function CoverageAttention:_buildModel(dim, coverageDim)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- previous hidden layer
  table.insert(inputs, nn.Identity()()) -- Context matrix batchL x sourceL x dim
  table.insert(inputs, nn.Identity()()) -- coverage vector batchL x sourceL x covDim

  local targetT = nn.Linear(dim, dim, false)(inputs[1]) -- batchL x dim
  local context = inputs[2] -- batchL x sourceTimesteps x dim
  
  local transformedCoverage = onmt.SequenceLinear(coverageDim, dim, false)(inputs[3]) -- no bias here 
  
  -- update the context matrix with the coverage vector
  local coveragedContext = nn.CAddTable()({context, transformedCoverage})
  
  -- Get attention.
  local attn = nn.MM()({coveragedContext, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
  attn = nn.Sum(3)(attn)
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  local alignmentVector = softmaxAttn(attn)
  attn = nn.Replicate(1,2)(alignmentVector) -- batchL x 1 x sourceL

  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x dim
  local contextVector = nn.Sum(2)(contextCombined) -- batchL x dim
  
  contextCombined = nn.JoinTable(2)({contextVector, inputs[1]})
  
  -- Learn a gate to softly control the flow between context and dec hidden
  local contextGate = nn.Sigmoid()(nn.Linear(dim*2, dim, true)(contextCombined))
  local inputGate = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(contextGate))
  
  local gatedContext = nn.CMulTable()({contextGate, contextVector})
  local gatedInput   = nn.CMulTable()({inputGate, inputs[1]})
  
  local gatedContextCombined = nn.JoinTable(2)({gatedContext, gatedInput})
  
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(gatedContextCombined))
  
  -- Also reupdate the coverage vector
  
  local newCoverage = onmt.ContextCoverage(dim, coverageDim)({inputs[3], inputs[2], alignmentVector})

  return nn.gModule(inputs, {contextOutput, newCoverage})
end
