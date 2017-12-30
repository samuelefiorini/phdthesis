#!/bin/bash

kill -9 `ps aux | grep samu | grep python | grep stability_selection | awk '{print $2}'` 2> /dev/null
echo "Output of ps aux | grep jdoe | grep pd_run :"
ps aux | grep samu | grep stability_selection
