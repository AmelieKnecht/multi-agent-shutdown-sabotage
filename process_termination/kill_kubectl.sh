#!/usr/bin/env bash
INSTANCE="__NAME__"
kubectl delete pods --namespace fleet --selector "instance=$INSTANCE" --grace-period=0
