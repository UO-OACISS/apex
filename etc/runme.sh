#!/bin/bash
# This script is executed by the bash shell which will initialize all
# environment variables as usual.

export TAU_ROOT=/opt/tau/tau2
export PATH=$TAU_ROOT/x86_64/bin:$PATH
cd /opt/tau/hpx.apex.tau/bin

rm -f profile.*

tau_exec -T pdt,pthread,serial $*
