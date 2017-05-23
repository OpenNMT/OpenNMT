local VDropout, Parent = torch.class('onmt.VDropout', 'nn.Module')

function VDropout:__init(p, max_batch_size, layer_size)
   print(max_batch_size,layer_size)
  Parent.__init(self)
  self.p = p or 0.5
  self.train = true
  if self.p >= 1 or self.p < 0 then
    error('<Dropout> illegal percentage, must be 0 <= p < 1')
  end
  self.noiseInit = torch.Tensor(1):zero()
  -- need to dimension it, otherwise won't keep its sharedness through resizing
  self.sharedNoise = torch.Tensor(max_batch_size, layer_size)
end


function VDropout:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  if self.p > 0 then
    if self.train then
      self.sharedNoise:resizeAs(input)
      if self.noiseInit[1]==0 then
         self.sharedNoise:bernoulli(1-self.p)
         self.sharedNoise:div(1-self.p)
         self.noiseInit[1] = 1
      end
      self.output:cmul(self.sharedNoise)
    end
  end
  return self.output
end

function VDropout:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  if self.train then
    if self.p > 0 then
      self.gradInput:cmul(self.sharedNoise) -- simply mask the gradients with the sharedNoise vector
    end
  end
  return self.gradInput
end

function VDropout:setp(p)
  self.p = p
end

function VDropout:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.p)
end


function VDropout:clearState()
  if self.sharedNoise then
    self.noiseInit[1] = 0
    self.sharedNoise:set()
  end
  return Parent.clearState(self)
end
