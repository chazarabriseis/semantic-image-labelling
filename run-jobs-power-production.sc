#!/bin/bash

input=$1
GPU=$2

loc=`pwd`


for i in {"48","52","56"}
do
	:
	name=`echo $input |sed 's/.py//g'`
	name=$name$i
	empty=" "
	echo $name
	inputint=$input$empty$i
	echo $inputint
	jbsub -queue p8 -cores 1x1+$GPU -name $name -out $name-out.txt -err $name-err.txt $HOME/test.sc $loc/$inputint
done
