
oldtag=v2.0
tagname=v2.0.1

echo "Commit Log:" >> ${tagname}.txt

git log ${oldtag}..HEAD --pretty=format:'http://github.com/khuck/xpress-apex/commit/%H %ad %s ' --reverse >> ${tagname}.txt

vi ${tagname}.txt

# git tag -a -F ${tagname}.txt ${tagname}