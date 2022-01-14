#!/bin/bash

binutils_path=$1

if [ ! -d ${binutils_path}/lib64 ] ; then
    ln -s ${binutils_path}/lib ${binutils_path}/lib64
fi
