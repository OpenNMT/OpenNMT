require 'nngraph'

local LSTM, parent = torch.class('LSTM', 'nn.Module')

function LSTM:__init(input_size, hidden_size)
  parent.__init(self)
  self.net = self:_buildModel(input_size, hidden_size)
  -- keep visibility on submodules for apply function
  self.modules = { self.net }
end

--[[Create a nngraph template of one time-step of a single-layer LSTM.

Parameters:

  * `input_size` - input size
  * `hidden_size` - internal size

Returns: An nngraph unit mapping: ${(c_{t-1}, h_{t-1}, x_t) => (c_{t}, h_{t})}$

--]]
function LSTM:_buildModel(input_size, hidden_size)
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

  local gmodule = nn.gModule(inputs, {next_c, next_h})
  gmodule.name = 'lstm'
  return gmodule
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

function LSTM:parameters()
  return self.net:parameters()
end

