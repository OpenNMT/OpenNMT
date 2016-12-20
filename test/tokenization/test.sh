dir=test/tokenization
ret=0

for file in $dir/*.test; do
    filename=$(basename "$file")
    name="${filename%.*}"
    th tools/tokenize.lua -case_feature < $file >tmp 2>/dev/null
    diff tmp $dir/$name.expected > /dev/null
    res=$?
    if [ $res -ne 0 ]; then
        echo "$name test failed"
        ret=1
    fi
done

exit $ret
