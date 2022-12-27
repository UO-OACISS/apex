#!/bin/bash -x

# update all remotes
git fetch origin
git fetch github

# checkout origin/develop from git.nic.uoregon.edu
git checkout thread_stats

# merge the github PR request changes
git merge github/thread_stats

# push to origin
git push origin thread_stats
