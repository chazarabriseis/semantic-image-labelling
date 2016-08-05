#!/bin/bash

input=$1
GPU=$2
label=$3

loc=`pwd`

name=`echo $input |sed 's/.py//g'`
name=$name$label
jbsub -queue p8 -cores 1x1+$GPU -name $name -out $name-out.txt -err $name-err.txt $HOME/test.sc $loc/$input
