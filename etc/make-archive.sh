#!/bin/bash

archive=apex.tar.gz
rm -f $archive
git archive origin/master --prefix apex/ | gzip > $archive
tar -tvf $archive