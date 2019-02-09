
oldtag=v2.1.1
tagname=v2.1.2

echo "Commit Log:" >> ${tagname}.txt

git log ${oldtag}..HEAD --pretty=format:'<li> <a href="http://github.com/khuck/xpress-apex/commit/%H">view commit &bull;</a> %s</li> ' --reverse > ${tagname}.txt
# git log ${oldtag}..HEAD --pretty=format:'http://github.com/khuck/xpress-apex/commit/%H %ad %s ' --reverse >> ${tagname}.txt

vi ${tagname}.txt

# git tag -a -f -F ${tagname}.txt ${tagname}