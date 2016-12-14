require 'nn'

function createCriterion(output_size)
    local criterion = nn.ParallelCriterion(false)
    local weights = torch.ones(output_size)
    weights[1] = 0
    local nll = nn.ClassNLLCriterion(weights)
    nll.sizeAverage = false
    criterion:add(nll)
    return criterion
end
