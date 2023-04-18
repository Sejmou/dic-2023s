import json
import logging
import re
from collections import defaultdict

from mrjob.job import MRJob

logger = logging.getLogger(__name__)


class AmazonReviewsChiSquared(MRJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # number of reviews in the dataset
        self.n = None
        # set of stopwords
        self.stopwords = None
        # dictionary of category counts
        self.category_counts = None

    def mapper_init(self):
        # load stopwords from file into a set
        with open("stopwords.txt", "r") as f:
            self.stopwords = set(f.read().splitlines())

    def mapper(self, _, line):
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
        combined_values = defaultdict(int)
        for category, count in values:
            combined_values[category] += count
        for item in combined_values.items():
            yield key, item

    def reducer_init(self):
        with open('category_counts.json', "r") as f:
            self.category_counts = json.load(f)
        self.n = self.category_counts.pop('number_of_reviews')

    def reducer(self, key, values):
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
            yield category, (chi_squared, key)


if __name__ == '__main__':
    AmazonReviewsChiSquared.run()
