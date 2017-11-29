To enable further customization of OpenNMT training, it is possible to easily modify the default behaviour of some modules, to add some options or even to disable some other. This is done with hooks.

Hooks are defined by a lua file (let us call it `myhook.lua`) that is dynamically loaded by passing the option `-hook_file myhook` in the different tools.

These hook files should return a table defining some functions corresponding to hook entry points in the code.

For instance, let us consider the following hook file:

```
local unicode = require('tools.utils.unicode')

local function mytokenization(_, line)
  local tokens = {}
  for _, c, _ in unicode.utf8_iter(line) do
    table.insert(tokens, c)
  end
  return tokens
end

return {
  tokenize = mytokenization
}
```

