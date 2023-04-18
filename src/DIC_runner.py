import argparse
import heapq
import logging
from collections import defaultdict

from amazon_reviews_chi_squared import AmazonReviewsChiSquared

# configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_job_output(job, runner):
    # dictionary to store the 75 terms with the highest chi-squared value for each category
    # use a default dictionary to avoid having to check if a category is already in the dictionary
    terms_for_category = defaultdict(lambda: [])

    # datastructure to store unique terms
    unique_terms = set()

    # loop through the output of the job in the format category, chi-squared value, term
    # and extract the 75 terms with the highest chi-squared value for each category
    # and store them in a dictionary with the category as key and the list of terms as value using a heapq
    for category, (chi_squared_value, term) in job.parse_output(runner.cat_output()):

        # if the heap of terms for the category has more than 75 elements,
        # remove the term with the lowest chi-squared value
        if len(terms_for_category[category]) >= 75:
            heapq.heappushpop(terms_for_category[category], (chi_squared_value, term))
        else:
            # add the term to the heap of terms for the category
            heapq.heappush(terms_for_category[category], (chi_squared_value, term))

    # Sort the dictionary based on the key, i.e. the category names alphabetically
    terms_for_category = dict(sorted(terms_for_category.items()))

    # Print the list of terms and chi_squared value for each category sorted by chi_squared value in descending order
    # in the format "<category> term1:chi_squared1 term2:chi_squared2 ... term75:chi_squared75"
    for category in terms_for_category:
        print("<%s>" % category, end="")
        for chi_squared_value, term in sorted(terms_for_category[category], reverse=True):
            unique_terms.add(term)
            print(" %s:%f" % (term, chi_squared_value), end="")
        print()

    # Print the list of unique terms in the format "term1 term2 term3 ... termN" sorted alphabetically
    print(" ".join(sorted(unique_terms)))


if __name__ == '__main__':
    # create argument parser
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
    else:
        myJob = AmazonReviewsChiSquared(
            args=[args.input, '--hadoop-streaming-jar', str(args.hadoop_streaming_jar), '-r', str(args.r), '--n',
                  str(args.n)])

    # add stopwords.txt to the list of files of the job
    myJob.FILES = ["../data/stopwords.txt"]

    # run the job using the specified runner
    with myJob.make_runner() as runner:
        # run the job
        runner.run()

        # parse the output of the job
        parse_job_output(myJob, runner)
