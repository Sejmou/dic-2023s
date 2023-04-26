#!/bin/bash

# This script takes two arguments: a file name and a runner name
# It runs the runner on the file and outputs the results to a file

# If no arguments are given, it runs the default runner on the default file
# If one argument is given, it runs the default runner on the given file
# If two arguments are given, it runs the given runner on the given file
if [ $# -eq 0 ]; then
  # NOTE: path to full dataset on cluster: hdfs:///user/dic23_shared/amazon-reviews/full/reviewscombined.json
  file="hdfs:///user/dic23_shared/amazon-reviews/full/reviews_devset.json"
  runner="./runner.py"
elif [ $# -eq 1 ]; then
  file="$1"
  runner="./runner.py"
elif [ $# -eq 2 ]; then
  file="$1"
  runner="$2"
else
  echo "Too many arguments"
  exit 1
fi

time python "$runner" --hadoop-streaming-jar /usr/lib/hadoop/tools/lib/hadoop-streaming-3.3.4.jar -r hadoop "$file" >output_ex1.txt
