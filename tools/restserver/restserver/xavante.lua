--[[
This is a modified version of the Restserver here https://github.com/hishamhm/restserver

]]

--[[
Copyright © 2016 Hisham Muhammad.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

local restserver_xavante = {}

local xavante = require("xavante")
local wsapi = require("wsapi.xavante")

local function start(self, check_func, every_sec)
   local rules = {}
   for path, _ in pairs(self.config.paths) do
      -- TODO support placeholders in paths
      rules[#rules + 1] = {
         match = path,
         with = wsapi.makeHandler(self.wsapi_handler)
      }
   end

   -- HACK: There's no public API to change the server identification
   xavante._VERSION = "SGA"
   xavante.HTTP {
      server = {host = self.config.host or "*", port = self.config.port or 8080 },
      defaultHost = {
         rules = rules
      }
   }

   local ok, err = pcall(xavante.start, check_func, every_sec)

   if not ok then
      return nil, err
   end
   return true
end

function restserver_xavante.extend(self)
   self.start = start
end

return restserver_xavante

