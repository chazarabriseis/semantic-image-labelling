#!/bin/bash

input=$1
GPU=$2
#QUEUE=$3
label=$3
QUEUE='k80_short'
loc=`pwd`
path='../Output/'
## QUEUE should be k80_short but can be anything as long as the -require "ostype==RHEL7.2" is used, it just may take longer to launch
name=`echo $input |sed 's/.py//g'`
name=$path$name$label
jbsub -queue $QUEUE -cores 1x1+$GPU -name $name -out $name-out.txt -err $name-err.txt -require "ostype=RHEL7.2" $loc/x86-7.2.sc $loc/$input
