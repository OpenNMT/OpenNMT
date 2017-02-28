local CrayonLogger = torch.class('CrayonLogger')

local options = {
  {'-exp_host', '127.0.0.1', [[Crayon server IP]], {}},
  {'-exp_port', '8889', [[Crayon Server port]], {}},
  {'-exp', '', [[Crayon experiment name]], {}}
}

function CrayonLogger.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Crayon')
end

function CrayonLogger:__init(args)
  self.args = onmt.utils.ExtendedCmdLine.getModuleOpts(args, options)
  if args.exp ~= '' then
    self.host = self.args.exp_host
    self.port = self.args.exp_port
    
    local crayon = require("crayon")
    self.cc = crayon.CrayonClient(self.host, self.port)
    self.exp = self.cc:create_experiment(args.exp)
    self.on = true

    
  else
    self.on = false
  end
end

return CrayonLogger
