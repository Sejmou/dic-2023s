import json
import logging
import re
from collections import defaultdict

from mrjob.job import MRJob

logger = logging.getLogger(__name__)


class AmazonReviewsChiSquared(MRJob):
    """
    This job calculates the chi-squared statistic for each pair of category and term in the dataset.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the job.
        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)

        # number of reviews in the dataset
        self.n = None
        # set of stopwords
        self.stopwords = None
        # dictionary of category counts
        self.category_counts = None

    def jobconf(self):
        """
        Set the number of reducers.
        :return: A dictionary of job configuration options.
        """
        # set the number of reducers using the jobconf dictionary
        orig_jobconf = super().jobconf()
        jobconf = {'mapreduce.job.reduces': 7}
        jobconf.update(orig_jobconf)
        return jobconf

    def mapper_init(self):
        """
        Initialize the mapper.
        Load the stopwords from file.
        :return: Nothing.
        """
        # load stopwords from file into a set
        with open("stopwords.txt", "r") as f:
            self.stopwords = set(f.read().splitlines())

    def mapper(self, _, line):
        """
        Map each review to a set of unique terms.
        Emit the category and the term.
        :param _: A key.
        Unused.
        :param line: A single line of the input file.
        Represents a single review.
        :return: Yields the term and a tuple of the form (category, 1).
        """
        # load json data from line
        review = json.loads(line)

        # extract the review text and the product category
        text = review["reviewText"]
        category = review["category"]

        # compile the regular expression pattern for splitting text into tokens
        pattern = re.compile(r"[^a-zA-Z<>^|]+")

        # set to keep track of seen terms
        seen_terms = set()

        # iterate over filtered terms and emit unique ones with the category
        for term in (term.lower() for term in pattern.split(text) if
                     len(term) >= 2 and term.lower() not in self.stopwords):
            if term not in seen_terms:
                # mark term as seen
                seen_terms.add(term)

                # emit the category and the term
                yield term, (category, 1)

    def combiner(self, key, values):
        """
        Combine the values for each term.
        :param key: A term.
        :param values: List of tuples of the form (category, 1).
        :return: Yields the term and a tuple of the form (category, count).
        """
        combined_values = defaultdict(int)
        for category, count in values:
            combined_values[category] += count
        for item in combined_values.items():
            yield key, item

    def reducer_init(self):
        """
        Initialize the reducer.
        Load the category counts from the file.
        :return: Nothing.
        """
        with open('category_counts.json', "r") as f:
            self.category_counts = json.load(f)
        self.n = self.category_counts.pop('number_of_reviews')

    def reducer(self, key, values):
        """
        Calculate the chi-squared statistic for each term and category.
        :param key: A term.
        :param values: List of tuples of the form (category, count).
        :return: Yields None and a tuple of the form (category, chi-squared, term).
        """
        # count the number of occurrences of each category for the term
        category_counts_for_term = defaultdict(int)
        aggregate_count = 0
        for category, count in values:
            category_counts_for_term[category] += count
            aggregate_count += count

        for category, count in category_counts_for_term.items():
            a = count
            b = aggregate_count - count
            c = self.category_counts[category] - count
            d = self.n - a - b - c
            chi_squared = self.n * ((a * d - b * c) ** 2) / ((a + b) * (a + c) * (b + d) * (c + d))
            yield None, (category, chi_squared, key)


if __name__ == '__main__':
    AmazonReviewsChiSquared.run()
