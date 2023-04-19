#!/usr/bin/env bash

op=${1:-install} # or upgrade

cd "$(dirname "$0")"
ver=$(cat ./VERSION)
ns=gpt4all

helm3 "${op}" gpt4all-api -n "${ns}" \
    --set namespace="${ns}" \
    --set image.tag="${ver}" \
    ./helm-chart
