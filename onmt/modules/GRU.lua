require('nngraph')

--[[
Implementation of a single stacked-GRU step as
an nn unit.

      h^L_{t-1} --- h^L_t
                 |


                 .
                 |
             [dropout]
                 |
      h^1_{t-1} --- h^1_t
                 |
                 |
                x_t

Computes $$(h_{t-1}, x_t) => (h_{t})$$.

--]]
local GRU, parent = torch.class('onmt.GRU', 'onmt.Network')

--[[
Parameters:

  * `layers` - Number of layers
  * `inputSize` - Size of input layer
  * `hiddenSize` - Size of the hidden layers
  * `regularization` - the regularization type
  * `dropout` - Dropout rate to use (in $$[0,1]$$ range).
  * `residual` - Residual connections between layers (boolean)
  * `dropout_input` - if true, add a dropout layer on the first layer (useful for instance in complex encoders)
--]]
function GRU:__init(layers, inputSize, hiddenSize, regularization, dropout, residual, dropout_input)
  dropout = dropout or 0

  self.regularization = regularization
  self.dropout = dropout
  self.numEffectiveLayers = layers
  self.outputSize = hiddenSize

  parent.__init(self, self:_buildModel(layers, inputSize, hiddenSize, regularization, dropout, residual, dropout_input))
end

--[[ Stack the GRU units. ]]
function GRU:_buildModel(layers, inputSize, hiddenSize, regularization, dropout, residual, dropout_input)
  -- inputs: { prevOutput L1, ..., prevOutput Ln, input }
  -- outputs: { output L1, ..., output Ln }

  local inputs = {}
  local outputs = {}

  for _ = 1, layers do
    table.insert(inputs, nn.Identity()()) -- h0: batchSize x hiddenSize
  end

  table.insert(inputs, nn.Identity()()) -- x: batchSize x inputSize
  local x = inputs[#inputs]

  local prevInput
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

    if regularization == 'dropout' and dropout > 0 then
      if (dropout_input or L > 1) then
        input = nn.Dropout(dropout)(input)
      end
    end

    local prevH = inputs[L]

    nextH = self:_buildLayer(inputDim, hiddenSize, regularization == 'layernorm')({prevH, input})
    prevInput = input

    table.insert(outputs, nextH)
  end

  return nn.gModule(inputs, outputs)
end

--[[ Build a single GRU unit layer.
    .. math::

            \begin{array}{ll}
            r_t = sigmoid(W_{xr} x_t + b_{xr} + W_{hr} h_{(t-1)} + b_{hr}) \\
            i_t = sigmoid(W_{xi} x_t + b_{xi} + W_hi h_{(t-1)} + b_{hi}) \\
            n_t = \tanh(W_{xn} x_t + b_{xn} + r_t * (W_{hn} h_{(t-1)} + b_{hn}) \\
            h_t = (1 - i_t) * n_t + i_t * h_{(t-1)} = n_t + i_t * (h_{(t-1) - n}) \\
            \end{array}

    where $$h_t` is the hidden state at time `t`, $$x_t$$ is the hidden
    state of the previous layer at time `t` or $$input_t$$ for the first layer,
    and $$r_t$$, $$i_t$$, $$n_t$$ are the reset, input, and new gates, respectively.

    In the function: `prevH`=$$h_{(t-1}}$$, `nextH`=$$h_t$$, `r`=$$r_t$$, `i`=$$i_t$$,
    `n`=$$n_t$$.

    `x2h_r,x2h_i,x2h_n` are $$W_{xr},W_{xi},W_{xn}$$ and the biases $$b_{xr},b_{xi},b_{xn}$$.

    And `h2h_r,h2h_i,h2h_n` are $$W_{hr},W_{hi},W_{hn}$$ and the biases $$b_{hr},b_{hi},b_{hn}$$.


]]
function GRU:_buildLayer(inputSize, hiddenSize, layerNorm)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  -- Recurrent input.
  local prevH = inputs[1]
  -- Previous layer input.
  local x = inputs[2]

  -- Evaluate the input sums at once for efficiency.
  local x2h = nn.Linear(inputSize, 3 * hiddenSize)(x)
  local h2h = nn.Linear(hiddenSize, 3 * hiddenSize)(prevH)

  -- Extract Wxr.x+bir, Wxi.x+bxi, Wxn.x+bin.
  local x2h_reshaped = nn.Reshape(3, hiddenSize)(x2h)
  local x2h_r, x2h_i, x2h_n = nn.SplitTable(2)(x2h_reshaped):split(3)

  -- Extract Whr.x+bhr, Whi.x+bhi, Whn.x+bhn
  local h2h_reshaped = nn.Reshape(3, hiddenSize)(h2h)
  local h2h_r, h2h_i, h2h_n = nn.SplitTable(2)(h2h_reshaped):split(3)

  -- Decode the gates.
  local a_r = nn.CAddTable()({x2h_r, h2h_r})
  local a_i = nn.CAddTable()({x2h_i, h2h_i})

  if layerNorm then
    a_r = onmt.LayerNormalization(hiddenSize)(a_r)
    a_i = onmt.LayerNormalization(hiddenSize)(a_i)
  end

  local r = nn.Sigmoid()(a_r)
  local i = nn.Sigmoid()(a_i)
  local a_n = nn.CAddTable()({
    x2h_n, nn.CMulTable()({r, h2h_n})
  })
  if layerNorm then
    a_n = onmt.LayerNormalization(hiddenSize)(a_n)
  end
  local n = nn.Tanh()(a_n)

  -- Perform the GRU update.
  local nextH = nn.CAddTable()({
    n,
    nn.CMulTable()({i, nn.CAddTable()({prevH, nn.MulConstant(-1)(n)})})
  })

  return nn.gModule(inputs, {nextH})
end
