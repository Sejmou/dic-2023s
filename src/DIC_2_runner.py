import heapq
import json
import logging
from collections import defaultdict

from category_counter import CategoryCounter
from chi_squared import AmazonReviewsChiSquared

# configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_job_output(job, runner):
    """
    Parse the output of a job.
    :param job: The job to parse the output of.
    :param runner: The runner to use to parse the output.
    :return: Nothing.
    """
    # dictionary to store the 75 terms with the highest chi-squared value for each category
    # use a default dictionary to avoid having to check if a category is already in the dictionary
    terms_for_category = defaultdict(lambda: [])

    # datastructure to store unique terms
    unique_terms = set()

    # loop through the output of the job in the format category, chi-squared value, term
    # and extract the 75 terms with the highest chi-squared value for each category
    # and store them in a dictionary with the category as key and the list of terms as value using a heapq
    for _, (category, chi_squared_value, term) in job.parse_output(runner.cat_output()):

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
    # create the job instances
    job1 = CategoryCounter()
    job2 = AmazonReviewsChiSquared()

    # load the stopwords file, and the category counts file as input files for the jobs
    job2.FILES = ["../data/stopwords.txt", "../category_counts.json"]

    # run the job using the specified runner
    with job1.make_runner() as runner1:
        # run the job
        runner1.run()

        # save the output of the job to a dictionary
        category_counts = defaultdict(int)
        number_of_reviews = 0
        for key, value in job1.parse_output(runner1.cat_output()):
            category_counts[key] = value
            number_of_reviews += value
        category_counts["number_of_reviews"] = number_of_reviews

        # Save the category counts to a json file
        with open("category_counts.json", "w") as f:
            f.write(json.dumps(category_counts))

    with job2.make_runner() as runner2:
        # run the job
        runner2.run()

        # parse the output of the job and print the results
        parse_job_output(job2, runner2)
