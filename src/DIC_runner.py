from amazon_reviews_chi_squared import AmazonReviewsChiSquared

if __name__ == '__main__':
    # create job
    myJob = AmazonReviewsChiSquared()

    # configure logging
    myJob.set_up_logging()

    # run job
    with myJob.make_runner() as runner:
        runner.run()
        for key, value in myJob.parse_output(runner.cat_output()):
            print(key, value, "\n", end='')
