import argparse

from amazon_reviews_chi_squared import AmazonReviewsChiSquared

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    # add command line options specifying the location of the input file
    parser.add_argument('input', type=str, help='Location of the input file')
    # add command line options specifying the number of reviews in the dataset
    parser.add_argument('--n', type=int, required=True, help='Number of reviews in the dataset')
    # parse command line arguments
    args = parser.parse_args()

    # create job
    myJob = AmazonReviewsChiSquared(args=[args.input, '--n', str(args.n)])

    # configure logging
    myJob.set_up_logging()

    # run job
    with myJob.make_runner() as runner:
        runner.run()
        for key, value in myJob.parse_output(runner.cat_output()):
            print(key, value, "\n", end='')
