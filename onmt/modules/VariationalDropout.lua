local VariationalDropout, parent = torch.class('onmt.VariationalDropout', 'nn.Module')

function VariationalDropout:__init(p)
  parent.__init(self)
  self.p = p or 0.5
  self.train = true
  if self.p >= 1 or self.p < 0 then
    error('<dropout> illegal percentage, must be 0 <= p < 1')
  end
  self.noiseInit = torch.Tensor(1):zero()
  self.sharedNoise = torch.Tensor(1,1)
end

function VariationalDropout.initializeNetwork(net)
  net:apply(function(m)
    if m.noiseInit then
      m.noiseInit[1] = 0
    end
  end)
end

function VariationalDropout:updateOutput(input)
  self.output:resizeAs(input):copy(input)

  if self.p > 0 and self.train then
    self.sharedNoise:resizeAs(input)

    if self.noiseInit[1] == 0 then
      self.sharedNoise:bernoulli(1 - self.p)
      self.sharedNoise:div(1 - self.p)
      self.noiseInit[1] = 1
    end

    self.output:cmul(self.sharedNoise)
  end

  return self.output
end

function VariationalDropout:updateGradInput(_, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)

  if self.p > 0 and self.train then
    -- Simply mask the gradients with the sharedNoise vector.
    self.gradInput:cmul(self.sharedNoise)
  end

  return self.gradInput
end

function VariationalDropout:setp(p)
  self.p = p
end

function VariationalDropout:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.p)
end

function VariationalDropout:clearState()
  if self.sharedNoise then
    self.noiseInit[1] = 0
    self.sharedNoise:set()
  end

  return parent.clearState(self)
end
