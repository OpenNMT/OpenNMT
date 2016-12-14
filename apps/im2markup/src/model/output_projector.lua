require 'nn'

function createOutputUnit(input_size, output_size)
    local model = nn.Sequential()
    model:add(nn.Linear(input_size, output_size))
    model:add(nn.LogSoftMax())
    return model
end
