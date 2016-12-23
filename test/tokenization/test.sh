LUA=$(which luajit lua | head -n 1)

dir=test/tokenization
ret=0
count=0
err=0
delim="_"

for file in $dir/*.raw; do
    filename=$(basename "$file")
    test="${filename%.*}"

    # Extract test options.
    name=$(echo $test | cut -f1 -d$delim)
    mode=$(echo $test | cut -f2 -d$delim)
    sep_annotate=$(echo $test | cut -f3 -d$delim)
    case=$(echo $test | cut -f4 -d$delim)

    tokenize_opts="-mode $mode"
    detokenize_opts=""

    if [ $case = "true" ]; then
        tokenize_opts="$tokenize_opts -case_feature"
        detokenize_opts="-case_feature"
    fi

    if [ $sep_annotate = "marker" ]; then
        tokenize_opts="$tokenize_opts -sep_annotate"
    fi

    # Test tokenization.
    $LUA tools/tokenize.lua $tokenize_opts < $dir/$test.raw >tmp 2>/dev/null
    diff tmp $dir/$name.tokenized > /dev/null
    res=$?
    if [ $res -ne 0 ]; then
        echo "* $name tokenization test failed"
        ret=1
        err=$(($err + 1))
    fi

    # Test detokenization.
    $LUA tools/detokenize.lua $detokenize_opts < $dir/$name.tokenized >tmp 2>/dev/null
    if [ -f $dir/$name.detokenized ]; then
        diff tmp $dir/$name.detokenized > /dev/null
    else
        diff tmp $dir/$test.raw > /dev/null
    fi
    res=$?
    if [ $res -ne 0 ]; then
        echo "* $name detokenization test failed"
        ret=1
        err=$(($err + 1))
    fi

    count=$(($count + 2))
done

rm -f tmp

echo "Completed $count tests with $err failures."

exit $ret
