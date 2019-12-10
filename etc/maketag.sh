#!/bin/bash -e

git checkout master
git merge develop

if [ -z ${oldtag+x} ]; then
    echo "oldtag is unset";
    kill -INT $$
else
    echo "oldtag is set to '$oldtag'";
fi
if [ -z ${tagname+x} ]; then
    echo "tagname is unset";
    kill -INT $$
else
    echo "tagname is set to '$tagname'";
fi

echo "Commit Log:" >> ${tagname}.txt

git log ${oldtag}..HEAD --pretty=format:'<li> <a href="http://github.com/khuck/xpress-apex/commit/%H">view commit &bull;</a> %s</li> ' --reverse > ${tagname}.txt
# git log ${oldtag}..HEAD --pretty=format:'http://github.com/khuck/xpress-apex/commit/%H %ad %s ' --reverse >> ${tagname}.txt

vi ${tagname}.txt

git tag -a -f -F ${tagname}.txt ${tagname}

git checkout develop