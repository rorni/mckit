#!/usr/bin/env sh

upstream_url="git@github.com:rorni/mckit.git"

git remote add upstream $upstream_url
git fetch --all
git merge upstream/master
# git push origin HEAD:master


# vim: set ts=4 sw=0 tw=79 ss=0 ft=sh et ai :
