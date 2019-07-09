#!/bin/bash -x

# update all remotes
git fetch origin
git fetch github

# checkout origin/develop from git.nic.uoregon.edu
git checkout develop

# merge the github PR request changes
git merge github/develop

# push to origin
git push origin develop
