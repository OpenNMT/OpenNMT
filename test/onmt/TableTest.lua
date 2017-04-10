require('onmt.init')

local tds = require('tds')

local tester = ...

local tableTest = torch.TestSuite()

function tableTest.subrange_all()
  local a = { 1, 2, 3, 4 }
  local b = onmt.utils.Table.subrange(a, 1, 4)
  tester:eq(b, a)
end

function tableTest.subrange_none()
  local a = { 1, 2, 3, 4 }
  local b = onmt.utils.Table.subrange(a, 2, 0)
  tester:eq(b, {})
end

function tableTest.subrange_some()
  local a = { 1, 2, 3, 4 }
  local b = onmt.utils.Table.subrange(a, 3, 2)
  tester:eq(b, { 3, 4 })
end


function tableTest.append_bothEmpty()
  local a = {}
  local b = {}
  onmt.utils.Table.append(a, b)
  tester:eq(a, {})
end

function tableTest.append_dstEmpty()
  local a = {}
  local b = { 1, 2 }
  onmt.utils.Table.append(a, b)
  tester:eq(a, b)
end

function tableTest.append_srcEmpty()
  local a = { 1, 2 }
  local b = {}
  onmt.utils.Table.append(a, b)
  tester:eq(a, { 1, 2 })
end

function tableTest.append_default()
  local a = { 1, 2 }
  local b = { 3, 4 }
  onmt.utils.Table.append(a, b)
  tester:eq(a, { 1, 2, 3, 4 })
end


function tableTest.merge_bothEmpty()
  local a = {}
  local b = {}
  onmt.utils.Table.merge(a, b)
  tester:eq(a, {})
end

function tableTest.merge_dstEmpty()
  local a = {}
  local b = { toto = 1, titi = 2 }
  onmt.utils.Table.merge(a, b)
  tester:eq(a, b)
end

function tableTest.merge_srcEmpty()
  local a = { toto = 1, titi = 2 }
  local b = {}
  local c = { toto = 1, titi = 2 }
  onmt.utils.Table.merge(a, b)
  tester:eq(a, c)
end

function tableTest.merge_default()
  local a = { toto = 1, titi = 2 }
  local b = { foo = 3, bar = 4 }
  local c = { toto = 1, titi = 2, foo = 3, bar = 4 }
  onmt.utils.Table.merge(a, b)
  tester:eq(a, c)
end


function tableTest.empty()
  tester:eq(onmt.utils.Table.empty({}), true)
  tester:eq(onmt.utils.Table.empty({ 1 }), false)
  tester:eq(onmt.utils.Table.empty({ toto = 1 }), false)
end


local function reorder(tester, tab, ind, exp, withTds)
  local res = onmt.utils.Table.reorder(tab, ind, withTds)
  tester:ne(torch.pointer(res), torch.pointer(tab))

  if withTds then
    tester:eq(torch.typename(res), 'tds.Vec')
    exp = tds.Vec(exp)
    local same = true
    for i = 1, #exp do
      if res[i] ~= exp[i] then
        same = false
        break
      end
    end
    tester:eq(same, true)
  else
    tester:eq(res, exp)
  end
end

function tableTest.reorder_empty()
  reorder(tester, {}, { 4, 3, 2, 1 }, {})
end
function tableTest.reorder_emptyWithTds()
  reorder(tester, {}, { 4, 3, 2, 1 }, {}, true)
end

function tableTest.reorder_reversed()
  reorder(tester, { 1, 2, 3, 4 }, { 4, 3, 2, 1 }, { 4, 3, 2, 1 })
end
function tableTest.reorder_reversedWithTds()
  reorder(tester, { 1, 2, 3, 4 }, { 4, 3, 2, 1 }, { 4, 3, 2, 1 }, true)
end


function tableTest.hasValue_empty()
  tester:eq(onmt.utils.Table.hasValue({}, 4), false)
end
function tableTest.hasValue_exists()
  tester:eq(onmt.utils.Table.hasValue({ 2, 6, 4, 9 }, 4), true)
end
function tableTest.hasValue_doesNotExist()
  tester:eq(onmt.utils.Table.hasValue({ 2, 6, 4, 9 }, 3), false)
end


return tableTest
