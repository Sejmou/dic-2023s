import argparse

from amazon_reviews_chi_squared import AmazonReviewsChiSquared

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # add command line options specifying the location of the hadoop streaming jar file
    parser.add_argument('--hadoop-streaming-jar', type=str, default=None,
                        help='Location of the hadoop streaming jar file')
    # add command line options specifying the type of runner to use for running the MapReduce job
    parser.add_argument('-r', type=str, default=None,
                        help='Specifies the type of runner to use for running the MapReduce job')
    # add command line options specifying the location of the input file
    parser.add_argument('input', type=str, help='Location of the input file')
    # add command line options specifying the number of reviews in the dataset
    parser.add_argument('--n', type=int, required=True, help='Number of reviews in the dataset')
    # parse command line arguments
    args = parser.parse_args()

    # create job
    if args.hadoop_streaming_jar is None and args.r is None:
        myJob = AmazonReviewsChiSquared(args=[args.input, '--n', str(args.n)])
        myJob.FILES = ["../data/stopwords.txt"]
    else:
        myJob = AmazonReviewsChiSquared(
            args=[args.input, '--hadoop-streaming-jar', str(args.hadoop_streaming_jar), '-r', str(args.r), '--n',
                  str(args.n)])
        myJob.FILES = ["stopwords.txt"]

    # configure logging
    myJob.set_up_logging(quiet=True)

    # run the job using the specified runner
    with myJob.make_runner() as runner:
        runner.run()
        counters = runner.counters()
        for key, value in myJob.parse_output(runner.cat_output()):
            if key is None:
                print(value, "\n", end='')
            else:
                print(key, value, "\n", end='')
