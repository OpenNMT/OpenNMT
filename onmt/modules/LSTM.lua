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
local LSTM, parent = torch.class('onmt.LSTM', 'nn.Container')

--[[
Parameters:

  * `layers` - Number of LSTM layers, $$L$$.
  * `input_size` - Size of input layer,  $$|x|$$.
  * `hidden_size` - Size of the hidden layers (cell and hidden, $$c, h$$).
  * `dropout` - Dropout rate to use.
  * `residual` - Residual connections between layers.
--]]
function LSTM:__init(layers, input_size, hidden_size, dropout, residual)
  parent.__init(self)

  dropout = dropout or 0

  self.dropout = dropout
  self.num_effective_layers = 2 * layers
  self.output_size = hidden_size

  self.net = self:_buildModel(layers, input_size, hidden_size, dropout, residual)
  self:add(self.net)
end

--[[ Stack the LSTM units. ]]
function LSTM:_buildModel(layers, input_size, hidden_size, dropout, residual)
  local inputs = {}
  local outputs = {}

  for _ = 1, layers do
    table.insert(inputs, nn.Identity()()) -- c0: batch_size x hidden_size
    table.insert(inputs, nn.Identity()()) -- h0: batch_size x hidden_size
  end

  table.insert(inputs, nn.Identity()()) -- x: batch_size x input_size
  local x = inputs[#inputs]

  local prev_input
  local next_c
  local next_h

  for L = 1, layers do
    local input
    local input_dim

    if L == 1 then
      -- First layer input is x.
      input = x
      input_dim = input_size
    else
      input_dim = hidden_size
      input = next_h
      if residual and (L > 2 or input_size == hidden_size) then
        input = nn.CAddTable()({input, prev_input})
      end
      if dropout > 0 then
        input = nn.Dropout(dropout)(input)
      end
    end

    local prev_c = inputs[L*2 - 1]
    local prev_h = inputs[L*2]

    next_c, next_h = self:_buildLayer(input_dim, hidden_size)({prev_c, prev_h, input}):split(2)
    prev_input = input

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  return nn.gModule(inputs, outputs)
end

--[[ Build a single LSTM unit layer. ]]
function LSTM:_buildLayer(input_size, hidden_size)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local prev_c = inputs[1]
  local prev_h = inputs[2]
  local x = inputs[3]

  -- Evaluate the input sums at once for efficiency.
  local i2h = nn.Linear(input_size, 4 * hidden_size)(x)
  local h2h = nn.Linear(hidden_size, 4 * hidden_size, false)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, hidden_size)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)

  -- Decode the gates.
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)

  -- Decode the write inputs.
  local in_transform = nn.Tanh()(n4)

  -- Perform the LSTM update.
  local next_c = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate, in_transform})
  })

  -- Gated cells form the output.
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return nn.gModule(inputs, {next_c, next_h})
end

function LSTM:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function LSTM:updateGradInput(input, gradOutput)
  return self.net:updateGradInput(input, gradOutput)
end

function LSTM:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end
