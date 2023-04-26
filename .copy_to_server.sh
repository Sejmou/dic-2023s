#!/bin/bash
# a simple script that copies the content of the current directory (except hidden files and directories) to some path on a remote server via ssh (scp)
# usage: ./.copy_to_server.sh <remote_user_and_server> <path>
# example: ./.copy_to_server.sh dic23@iccluster118 dic23/exercise1

# check if the number of arguments is correct
if [ $# -ne 2 ]; then
  echo "Usage: copy_to_server.sh <remote_user_and_server> <path>"
  exit 1
fi

remote_user_and_server="$1"
path="$2"
script_name=$(basename "$0")

# copy the content of the current directory (except hidden files) to the remote server
# note: if file name of script starts without a dot, it will be copied as well!
scp -r [!.]* "$remote_user_and_server:$path"
