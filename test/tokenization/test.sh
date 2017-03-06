LUA=$(which luajit lua | head -n 1)

dir=test/tokenization
bpe_dir=bpe-models
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
    joiner_annotate=$(echo $test | cut -f3 -d$delim)
    case=$(echo $test | cut -f4 -d$delim)
    bpe=$(echo $test | cut -f5 -d$delim)

    tokenize_opts="-mode $mode"
    detokenize_opts=""

    if [ $case = "true" ]; then
        tokenize_opts="$tokenize_opts -case_feature"
        detokenize_opts="-case_feature"
    fi

    if [ $joiner_annotate = "marker" ]; then
        tokenize_opts="$tokenize_opts -joiner_annotate"
    fi

    if [ $bpe ]; then
        tokenize_opts="$tokenize_opts -bpe_model $dir/$bpe_dir/$bpe -EOT_marker </w> -bpe_mode suffix"
    fi

    # Test tokenization 1.
    $LUA tools/tokenize.lua $tokenize_opts < $dir/$test.raw >tmp 2>/dev/null
    diff tmp $dir/$name.tokenized > /dev/null
    res=$?
    if [ $res -ne 0 ]; then
        echo "* $name tokenization test failed"
        ret=1
        err=$(($err + 1))
    fi
    if [ -f $dir/$name.tokenized.new ]; then
      # Test tokenization 2.
      $LUA tools/tokenize.lua $tokenize_opts -joiner_new < $dir/$test.raw >tmp 2>/dev/null
      diff tmp $dir/$name.tokenized.new > /dev/null
      res=$?
      if [ $res -ne 0 ]; then
          echo "* $name tokenization (joiner new) test failed"
          ret=1
          err=$(($err + 1))
      fi
    fi

    # Test detokenization 1.
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
    if [ -f $dir/$name.tokenized.new ]; then
      # Test detokenization 2.
      $LUA tools/detokenize.lua $detokenize_opts < $dir/$name.tokenized.new >tmp 2>/dev/null
      if [ -f $dir/$name.detokenized ]; then
          diff tmp $dir/$name.detokenized > /dev/null
      else
          diff tmp $dir/$test.raw > /dev/null
      fi
      res=$?
      if [ $res -ne 0 ]; then
          echo "* $name detokenization (joiner_new) test failed"
          ret=1
          err=$(($err + 1))
      fi
      count=$(($count + 2))
    fi
    count=$(($count + 2))
done

rm -f tmp

echo "Completed $count tests with $err failures."

exit $ret
