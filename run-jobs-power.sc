#!/bin/bash

input=$1
GPU=$2

loc=`pwd`

name=`echo $input |sed 's/.py//g'`

jbsub -queue p8 -cores 1x1+$GPU -name $name -out $name-out.txt -err $name-err.txt $HOME/test.sc $loc/$input
#jbsub -queue p8 -cores 1x1+$GPU -name $name -out $name-out.txt -err $name-err.txt -mem 20g $HOME/test.sc $loc/$input
