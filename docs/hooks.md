To enable further customization of OpenNMT, it is possible to easily modify the default behaviour of some modules, to add some options or even to disable some others. This is done with `hooks`.

Hooks are defined by a lua file (let us call it `myhook.lua`) containing the extension code and that is dynamically loaded by passing the option `-hook_file myhook` in the different tools.

These hook files should return a table defining some functions corresponding to *hook entry points* in the code. These functions are replacing or introducing some code in the standard flow. These functions can also decide based on the actual request, to not change the standard flow, in which case, they have to return a `nil` value.

## Example

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

save it in `myhook.lua` and let us try a standard tokenization:

```
$ echo "c'est épatant" | th tools/tokenize.lua -hook_file myhook
c ' e s t _ é p a t a n t
Tokenization completed in 0.001 seconds - 1 sentences
```

the tokenization function has taken over the normal tokenization.

Let us do that more cleanly, first - document this as a new option, we just need to declare the new option in the hook file:

```
local myopt =
{
  {
    '-mode', 'conservative',
    [[Define how aggressive should the tokenization be. `aggressive` only keeps sequences
      of letters/numbers, `conservative` allows a mix of alphanumeric as in: "2,000", "E65",
      "soft-landing", etc. `space` is doing space tokenization. `char` is doing character tokenization]],
    {
      enum = {'space', 'conservative', 'aggressive', 'char'}
    }
  }
}
```

and define a hook to the `declareOpts` function:

```
local function declareOptsFn(cmd)
  cmd:setCmdLineOptions(myopt, 'Tokenizer')
end

return {
  tokenize = mytokenization,
  declareOpts = declareOptsFn
}
```

checking the help now shows the updated option:

```
th tools/tokenize.lua -hook_file myhook -h:

[...]
  -mode <string> (accepted: space, conservative, aggressive, char; default: conservative)
      Define how aggressive should the tokenization be. `aggressive` 
      only keeps sequences of letters/numbers, `conservative` allows 
      a mix of alphanumeric as in: "2,000", "E65", "soft-landing", 
      etc. `space` is doing space tokenization. `char` is doing character 
      tokenization
[...]
```

we just need to redefine only this new mode in the tokenization function, for that, we now check the variable `opt`:

```
local function mytokenization(opt, line)
  -- fancy tokenization, it has to return a table of tokens (possibly with features)
  if opt.mode == "char" then
    local tokens = {}
    for v, c, _ in unicode.utf8_iter(line) do
      if unicode.isSeparator(v) then
        table.insert(tokens, '_')
      else
        table.insert(tokens, c)
      end
    end
    return tokens
  end
end
```

which gives the regular behavior by default:

```
$ echo "c'est épatant" | th tools/tokenize.lua -hook_file myhook
gives the expected c ' est épatant
```

and the updated behavior with the hook and the option set:

```
$ echo "c'est épatant" | th tools/tokenize.lua -hook_file myhook -mode char
gives the new: c ' e s t _ é p a t a n t
```

## Sample Hooks

Several hooks are available in `hooks` directory - see description [here](https://github.com/OpenNMT/OpenNMT/blob/master/hooks/README.md).

## Testing hooks

You can run unit tests with hook by adding `-hook_file myhook` as a first parameter of test.lua:

```
th test/test.lua -hook_file myhook
```

It is a good practice to run the complete set of tests with your hooks to check for possible side-effects.

To add specific tests for your hook, you have to define a hook on `hookName` returning the name of the hook, and use this value to add dynamically additional tests. Look at `test/auto/TokenizerTest.lua` for an example:

```
function tokenizerTest.hooks()
  local hookName = _G.hookManager:call("hookName")
  if hookName == "chartok" then
    local opt = cmd:parse({'-mode', 'char'})
    testTok(opt, "49th meeting Social and human rights questions [14 (g)]", 
                 "4 9 t h ▁ m e e t i n g ▁ S o c i a l ▁ a n d ▁ h u m a n ▁ r i g h t s ▁ q u e s t i o n s ▁ [ 1 4 ▁ ( g ) ]")
  elseif hookName == "sentencepiece" then
    local opt = cmd:parse({'-mode','none', '-sentencepiece' ,'hooks/lua-sentencepiece/test/sample.model'})
    testTok(opt, "une impulsion Berry-Siniora pourraient changer quoi", 
                 "▁un e ▁imp ul s ion ▁B erry - S i nior a ▁po ur ra i ent ▁change r ▁ quoi")
  end
end
``` 

## Predefined hook entry point

Special entry points:

* `declareOpts`: this hook is in charge of modifying the actual option for the current script - for instance, it is possible to add, remove or modify some options. A hook does not necessarily require new options - you can defined a hook that operates by default to replace internal features by a more efficient implementation. To avoid confusion, make sure in these cases that all the options are supported and the result is the same.
* `hookName`: special hook, allowing the code to know about the name of the current hook. `function hookName() return NAME end` - can be used in automatic tests

Normal entry points:

* `tokenize`: replace or extend internal tokenization - prototype is: `function mytokenization(opt, line, bpe)` that has to return a table of tokens (and possible features). See [hooks/chartokenization.lua](https://github.com/OpenNMT/OpenNMT/blob/master/hooks/chartokenization.lua) for an example.
* `detokenize`: replace or extend internal detokenization - prototype is: `function mytokenization(line, opt)` that has to return a detokenized string. See [hooks/chartokenization.lua](https://github.com/OpenNMT/OpenNMT/blob/master/hooks/chartokenization.lua) for an example.
* `post_tokenize`: performs a transformation of the tokens list just after tokenization. Typically interesting to add features. See [[hooks/chartokenization.lua](https://github.com/OpenNMT/OpenNMT/blob/master/hooks/treetagger-tag.lua) for an example.
* `mpreprocess`: on the fly normalization on source and target strings during preprocessing. `function mpreprocessFn(opt, line)` that returns normalized line. Note that the options associated to the source or target options are dynamically changed for the call.
* `bpreprocess`: performs a bilingual normalization of the source and target sentences during preprocessing. `function bpreprocessFn(opt, sentences)`. `sentences` is a `{srcs,tgts}` where `srcs` and `tgts` are respectively list of source and target sentences. Sentences are sent by batch of 10000.
