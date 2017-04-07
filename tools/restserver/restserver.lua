
local restserver = {}

local request = require("wsapi.request")
local response = require("wsapi.response")
local json = require("dkjson")
local unpack = unpack or table.unpack

local function add_resource(self, name, entries)
   for _, entry in ipairs(entries) do
      local path = ("/" .. name .. "/" .. entry.path):gsub("/+", "/"):gsub("/$", "")
      entry.rest_path = path
      entry.match_path = path:gsub("{[^:]*:([^}]*)}", "(%1)"):gsub("{[^}]*}", "([^/]+)") .. "$"
      path = path:gsub("{[^:]*:([^}]*)}", "%1"):gsub("{[^}]*}", "[^/]+") .. "$"
      local methods = self.config.paths[path]
      if not methods then
         methods = {}
         self.config.paths[path] = methods
         table.insert(self.config.path_list, path)
      end
--      if methods[entry.method] then
--         error("A handler for method "..entry.method.." in path "..path.." is already defined.")
--      end
      methods[entry.method] = entry
   end
end


local function type_check(tbl, schema)
   for k, s in pairs(schema) do
      if not tbl[k] and not s.optional then
         return nil, "missing field '"..k.."'"
      elseif type(tbl[k]) ~= s.type then
         return nil, "in field '"..k.."', expected type "..s.type..", got "..type(tbl[k])
      elseif s.array and next(tbl[k]) and not tbl[k][1] then
         return nil, "in field '"..k.."', expected an array"
      end
   end
   return true
end

local function decode(data, mimetype, schema)
   if mimetype == "application/json" then
      local tbl = json.decode(data)
      if schema then
         local ok, err = type_check(tbl, schema)
         if not ok then
            return nil, err
         end
      end
      return tbl
   elseif mimetype == "text/plain" then
      return data or ""
   elseif not mimetype or mimetype == "*/*" then
      return data or ""
   else
      error("Mimetype "..mimetype.." not supported.")
   end
end

local function encode(data, mimetype, schema)
   if mimetype == "application/json" then
      if schema then
         local ok, err = type_check(data, schema)
         if not ok then
            return nil, err
         end
      end
      return json.encode(data)
   elseif mimetype == "text/plain" then
      return data or ""
   elseif not mimetype then
      return data or ""
   else
      error("Mimetype "..mimetype.." not supported.")
   end
end

local function fail(code, msg)
   local wres = response.new(code, { ["Content-Type"] = "text/plain" })
   wres:write(tostring(code).." "..msg)
   return wres:finish()
end

local function match_path(self, path_info)
   for _, path in ipairs(self.config.path_list) do
      if path_info:match(path) then
         return self.config.paths[path]
      end
   end
end

local function wsapi_handler_with_self(self, wsapi_env)
   local wreq = request.new(wsapi_env)
   local methods = self.config.paths[wsapi_env.PATH_INFO] or match_path(self, wsapi_env.PATH_INFO)

   local entry = methods and methods[wreq.method]
   if not entry then
      return fail(405, "Method Not Allowed")
   end

   local input, err
   if wreq.method == "POST" then
      input, err = decode(wreq.POST.post_data, entry.consumes, entry.input_schema)
   elseif wreq.method == "GET" then
      input = wreq.GET
   elseif wreq.method == "DELETE" then
      input = ""
   else
      error("Other methods not implemented yet.")
   end
   if not input then
      return fail(400, "Bad Request - Your request fails schema validation: "..err)
   end

   local placeholder_matches = (entry.rest_path ~= entry.match_path) and { wsapi_env.PATH_INFO:match(entry.match_path) } or {}
   local ok, res = pcall(entry.handler, input, unpack(placeholder_matches))
   if not ok then
      return fail(500, "Internal Server Error - Error in application: "..res)
   end
   if not res then
      return fail(500, "Internal Server Error - Server failed to produce a response.")
   end

   local output, err = encode(res.config.entity, entry.produces, entry.output_schema)
   if not output then
      return fail(500, "Internal Server Error - Server built a response that fails schema validation: "..err)
   end
   
   local wres = response.new(res.config.status, { ["Content-Type"] = entry.produces or "text/plain" })
   wres:write(output)
   return wres:finish()
end

local function add_setter(self, var)
   self[var] = function (self, val)
      self.config[var] = val
      return self
   end
end

function restserver.new(options)
   local server = {
      config = {
         paths = {},
         path_list = {},
      },
      enable = function(self, plugin_name)
         local mod = require(plugin_name)
         mod.extend(self)
         return self
      end,
      add_resource = add_resource,
      timer = {},
      elapsed = {},
      model_loaded = {},
      translator = {},
      opt = {},
   }
   add_setter(server, "host")
   add_setter(server, "port")
   server.wsapi_handler = function(wsapi_env)
      return wsapi_handler_with_self(server, wsapi_env)
   end
   return server
end

function restserver.response()
   local res = {
      config = {},
   }
   add_setter(res, "status")
   add_setter(res, "entity")
   return res
end

return restserver
