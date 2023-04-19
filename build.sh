#!/usr/bin/env bash

cd "$(dirname "$0")"
ver=$(cat ./VERSION)
cr="gcr.io/apt-phenomenon-243802"

docker build . -t "${cr}/gpt4all-api:${ver}" -f ./docker/Dockerfile
docker push "${cr}/gpt4all-api:${ver}"

cd -