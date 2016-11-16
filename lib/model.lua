require 'torch'

local Model = torch.class('Model')

function Model:__init()
end

function Model:double()
  self:convert(function (obj)
    return obj:double()
  end)
end

function Model:float()
  self:convert(function (obj)
    return obj:float()
  end)
end

function Model:cuda()
  self:convert(function (obj)
    return obj:cuda()
  end)
end
