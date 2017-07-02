require('nngraph')

--[[
Implementation of a single stacked-LSTM step as
an nn unit.

      h^L_{t-1} --- h^L_t
      c^L_{t-1} --- c^L_t
                 |


                 .
                 |
             [dropout]
                 |
      h^1_{t-1} --- h^1_t
      c^1_{t-1} --- c^1_t
                 |
                 |
                x_t

Computes $$(c_{t-1}, h_{t-1}, x_t) => (c_{t}, h_{t})$$.

--]]
local LSTM, parent = torch.class('onmt.LSTM', 'onmt.Network')

--[[
Parameters:

  * `layers` - Number of LSTM layers, L.
  * `inputSize` - Size of input layer
  * `hiddenSize` - Size of the hidden layers.
  * `dropout` - Dropout rate to use (in $$[0,1]$$ range).
  * `residual` - Residual connections between layers.
  * `dropout_input` - if true, add a dropout layer on the first layer (useful for instance in complex encoders)
  * `dropout_type` - naive dropout applies independently of each connection, variational applies uniformally on all timesteps
--]]
function LSTM:__init(layers, inputSize, hiddenSize, dropout, residual, dropout_input, dropout_type)
  dropout = dropout or 0

  self.dropout = dropout
  self.numStates = 2 * layers
  self.outputSize = hiddenSize

  parent.__init(self, self:_buildModel(layers, inputSize, hiddenSize, dropout, residual, dropout_input, dropout_type))
end

--[[ Stack the LSTM units. ]]
function LSTM:_buildModel(layers, inputSize, hiddenSize, dropout, residual, dropout_input, dropout_type)
  local inputs = {}
  local outputs = {}

  for _ = 1, layers do
    table.insert(inputs, nn.Identity()()) -- c0: batchSize x hiddenSize
    table.insert(inputs, nn.Identity()()) -- h0: batchSize x hiddenSize
  end

  table.insert(inputs, nn.Identity()()) -- x: batchSize x inputSize
  local x = inputs[#inputs]

  local prevInput
  local nextC
  local nextH

  for L = 1, layers do
    local input
    local inputDim

    if L == 1 then
      -- First layer input is x.
      input = x
      inputDim = inputSize
    else
      inputDim = hiddenSize
      input = nextH
      if residual and (L > 2 or inputSize == hiddenSize) then
        input = nn.CAddTable()({input, prevInput})
      end
    end

    local prevC = inputs[L*2 - 1]
    local prevH = inputs[L*2]

    -- Apply variational dropout on recurrent connection.
    if dropout_type == "variational" then
      prevH = onmt.VariationalDropout(dropout)(prevH)
    end
    if dropout_input or L > 1 then
      if dropout_type == "variational" then
        input = onmt.VariationalDropout(dropout)(input)
      else
        input = nn.Dropout(dropout)(input)
      end
    end

    nextC, nextH = self:_buildLayer(inputDim, hiddenSize)({prevC, prevH, input}):split(2)
    prevInput = input

    table.insert(outputs, nextC)
    table.insert(outputs, nextH)
  end

  return nn.gModule(inputs, outputs)
end

--[[ Build a single LSTM unit layer. ]]
function LSTM:_buildLayer(inputSize, hiddenSize)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local prevC = inputs[1]
  local prevH = inputs[2]
  local x = inputs[3]

  -- Evaluate the input sums at once for efficiency.
  local i2h = nn.Linear(inputSize, 4 * hiddenSize)(x)
  local h2h = nn.Linear(hiddenSize, 4 * hiddenSize)(prevH)
  local allInputSums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, hiddenSize)(allInputSums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

  -- Decode the gates.
  local inGate = nn.Sigmoid()(n1)
  local forgetGate = nn.Sigmoid()(n2)
  local outGate = nn.Sigmoid()(n3)

  -- Decode the write inputs.
  local inTransform = nn.Tanh()(n4)

  -- Perform the LSTM update.
  local nextC = nn.CAddTable()({
    nn.CMulTable()({forgetGate, prevC}),
    nn.CMulTable()({inGate, inTransform})
  })

  -- Gated cells form the output.
  local nextH = nn.CMulTable()({outGate, nn.Tanh()(nextC)})

  return nn.gModule(inputs, {nextC, nextH})
end
