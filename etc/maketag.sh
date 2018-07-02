
tagname=v2.0.1

echo "Commit Log:" >> ${tagname}.txt

git log $(git describe --tags --abbrev=0)..HEAD --pretty=format:'http://github.com/khuck/xpress-apex/commit/%H %ad %s ' --reverse >> ${tagname}.txt

vi ${tagname}.txt

# git tag -a -F ${tagname}.txt ${tagname}