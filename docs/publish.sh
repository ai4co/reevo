#!/bin/bash
set -e

tmpfolder=/tmp/reevo-gh-pages
commit_hash="$(git rev-parse --short HEAD)"
origin_url="$(git remote get-url origin)"

rm -rf "${tmpfolder}"
git clone "${origin_url}" -b gh-pages --single-branch "${tmpfolder}"
rm -r "${tmpfolder}"/*
python3 build.py
cp -r ./dist/* "${tmpfolder}/"
cd "${tmpfolder}"

git add -A
git commit -m "Build website ($(date "+%F %T"), ${commit_hash})"
git push origin gh-pages:gh-pages