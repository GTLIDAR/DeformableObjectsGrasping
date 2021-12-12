#!/bin/sh

a=10

while [ $a -lt 90 ]
do 
	echo $a
	src="validation/object0"$a"_result.dat"
	dst="Random/object0"$a"_result.dat"
	dst_rm="Random/object00"$a"_result.dat"
	rm $dst_rm
	cat $src >> $dst
	a=`expr $a + 1`
done
