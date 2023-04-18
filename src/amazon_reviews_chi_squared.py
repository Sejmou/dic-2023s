import json
import logging
import re
from collections import defaultdict

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.util import log_to_stream

logger = logging.getLogger(__name__)


class AmazonReviewsChiSquared(MRJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # number of reviews in the dataset
        self.n = None

        # set of stopwords
        self.stopwords = None

        # encodings for the document count for a category
        self.document_count_for_category_encoding = 'n'
        # encodings for the number of documents in category which contain term
        self.a_encoding = 'A'
        # encodings for the number of documents not in category which contain term
        self.b_encoding = 'B'
        # encodings for the number of documents in category which do not contain term
        self.c_encoding = 'C'

    def configure_args(self):
        # add command line options specifying the number of reviews in the dataset
        super().configure_args()
        self.add_passthru_arg(
            '--n',
            dest='n',
            type=int,
            help='number of reviews in the dataset')

    def set_up_logging(cls, quiet=False, verbose=False, stream=None):
        log_to_stream(name="mrjob", debug=verbose, stream=stream)
        log_to_stream(name="__main__", debug=verbose, stream=stream)
        log_to_stream(name="amazon_reviews_chi_squared", debug=verbose, stream=stream)

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
                yield category, (term, 1)
                # emit the category and the document count
                yield term, (category, 1)

        # emit the category and the document count
        yield category, (self.document_count_for_category_encoding, 1)

    def combiner(self, key, values):
        # combine the values for each key
        combined_values = defaultdict(int)

        # iterate over the values and combine them
        for term, count in values:
            combined_values[term] += count

        # emit the combined values
        for item in combined_values.items():
            yield key, item

    def reducer_calculate_variables(self, key, values):
        # Check if the key is a category (i.e. if key[0].isupper())
        if key[0].isupper():
            category = key
            # count the number of occurrences of each term in the category
            term_counts_for_category = defaultdict(int)
            for term, count in values:
                term_counts_for_category[term] += count

            # extract the number of documents in the category
            number_of_documents_for_category = term_counts_for_category.pop(self.document_count_for_category_encoding)

            for term, count in term_counts_for_category.items():
                # emit the number of documents in category which contain term
                yield (category, term), (self.a_encoding, count)
                # emit the number of documents in category which do not contain term
                yield (category, term), (self.c_encoding, number_of_documents_for_category - count)

        # otherwise, the key is a term
        else:
            term = key
            if term == self.document_count_for_category_encoding:
                return

            # count the number of occurrences of each category for the term
            category_counts_for_term = defaultdict(int)
            aggregate_count = 0
            for category, count in values:
                category_counts_for_term[category] += count
                aggregate_count += count

            for category, count in category_counts_for_term.items():
                # emit the number of documents not in category which contain term
                yield (category, term), (self.b_encoding, aggregate_count - count)

    def reducer_chi_squared(self, key, values):
        # extract the category and the term
        category, term = key

        n = self.options.n
        a, b, c = 0, 0, 0
        # extract the values for a, b, and c
        for encoding, value in values:
            if encoding == self.a_encoding:
                a = value
            elif encoding == self.b_encoding:
                b = value
            elif encoding == self.c_encoding:
                c = value

        # calculate d, the number of documents not in category which do not contain term
        d = n - a - b - c

        # calculate chi-squared statistic for the term in the category
        chi_squared = n * ((a * d - b * c) ** 2) / ((a + b) * (a + c) * (b + d) * (c + d))

        yield category, (chi_squared, term)

    def steps(self):
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                combiner=self.combiner,
                reducer=self.reducer_calculate_variables,
            ),
            MRStep(
                reducer=self.reducer_chi_squared,
            )
        ]


if __name__ == '__main__':
    AmazonReviewsChiSquared.run()
