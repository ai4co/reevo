#!/bin/bash
set -e

tmpfolder=/tmp/reevo-gh-pages
code_hash="$(git rev-parse HEAD)"

git clone ssh://git@ssh.github.com:443/ai4co/reevo.git -b gh-pages --single-branch "${tmpfolder}"
rm -r "${tmpfolder}"/*
python3 build.py
cp -r ./dist/* "${tmpfolder}/"
cd "${tmpfolder}"

git add -A
git commit -m "Build website ($(date "+%F %T"), ${code_hash})"
git push origin gh-pages:gh-pages