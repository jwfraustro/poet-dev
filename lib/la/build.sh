#!/bin/bash

#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.

#this will remove any partial builds, if there was a failure

#to build in python 3, use "build.sh 3"


if [ "$1" == "clean" ]
then
    rm least_asymmetry/*.so
    exit
fi

"python$1" setup.py build_ext --inplace 
#rm -r build/
"python$1" unit_test.py
echo "~~You made it!~~"
