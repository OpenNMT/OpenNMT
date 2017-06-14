# Core

## Adding new tests

Add a new file under `test/onmt/` following this template:

```lua
-- The test manager `tester` is passed by the calling script.
local tester = ...

local myTests = torch.TestSuite()

function myTests.trueIsTrue()
  tester:eq(true, true)
end

return myTests
```

It will be automatically loaded by the test script.

## Running tests

1. Go to the top-level OpenNMT directory
2. `th test/test.lua`

You can run a subset of the test suite by passing a list of test name patterns. For example:

```
th test/test.lua Batch Dataset
```

will run tests containing `Batch` and `Dataset` in their name.
