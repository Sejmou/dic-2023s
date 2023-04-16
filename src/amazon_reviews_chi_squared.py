import heapq
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
        self.n = None
        self.stopwords = None
        self.category_encoding = 'c'
        self.document_count_for_category_encoding = 'n'
        self.term_encoding = 't'
        self.a_encoding = 'A'
        self.b_encoding = 'B'

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

        # iterate over tokens in text
        for token in set(pattern.split(text)):
            # case folding
            token = token.lower()

            # skip terms with less than 2 characters
            if len(token) < 2:
                continue

            # skip stopwords
            if token in self.stopwords:
                continue

            # emit the term and the category
            yield (token, self.term_encoding), (category, 1)

            # emit the category and the term
            yield (category, self.category_encoding), (token, 1)

        # emit the category and the document count
        yield (category, self.category_encoding), (self.document_count_for_category_encoding, 1)

    def combiner(self, key, values):
        # combine the values for the same key
        combined_counts = defaultdict(int)
        # for value in values:
        for value, _ in values:
            combined_counts[value] += 1

        # emit the combined values
        for item in combined_counts.items():
            yield key, item

    def reducer_calculate_variables(self, key, values):
        # extract the encoding
        _, encoding = key

        if encoding == self.category_encoding:
            # extract the category
            category, _ = key

            # count the number of occurrences of each term in the category
            term_counts_for_category = defaultdict(int)
            for term, count in values:
                term_counts_for_category[term] += count

            # emit the number of documents in the category
            category_count = term_counts_for_category.pop(self.document_count_for_category_encoding)
            yield category, (self.document_count_for_category_encoding, None, category_count)

            # emit the term count for the category
            for term, count in term_counts_for_category.items():
                yield category, (self.a_encoding, term, count)

        elif encoding == self.term_encoding:
            # extract the term
            term, _ = key

            # emit the term needed for the listing of all terms
            yield None, term

            # count the number of occurrences of each category for the term
            category_counts_for_term = defaultdict(int)
            aggregate_count = 0
            for category, count in values:
                category_counts_for_term[category] += count
                aggregate_count += count

            # emit the number of documents not in the category for each term
            # aggregate_count = sum(category_counts_for_term.values())
            for category, count in category_counts_for_term.items():
                yield category, (self.b_encoding, term, aggregate_count - count)

    def reducer_chi_squared(self, key, values):
        # Check if the key is None (i.e. the term list)
        if key is None:
            # emit a space-separated list of all terms
            yield None, " ".join(sorted(values))
            # terminate the reducer function
            return

        # create a dictionary for each term
        term_dict = defaultdict(lambda: {"a": 0, "b": 0})
        # initialize the number of documents in the category
        category_count = 0

        # extract A, B, values for each term and the number of documents in the category
        for (encoding, term, value) in values:
            # extract the number of documents in the category for each term
            if encoding == self.a_encoding:
                term_dict[term]["a"] = value
            # extract the number of documents not in the category for each term
            elif encoding == self.b_encoding:
                term_dict[term]["b"] = value
            # extract the number of documents in the category
            elif encoding == self.document_count_for_category_encoding:
                category_count = value

        # calculate chi squared for each term
        top_terms = []
        n = self.options.n
        for term, term_values in term_dict.items():
            a = term_values["a"]
            b = term_values["b"]
            c = category_count - a
            d = n - a - b - c

            # save the chi squared value into the dictionary
            chi_squared = n * ((a * d - b * c) ** 2) / ((a + c) * (b + d) * (a + b) * (c + d))

            # keep track of the top 75 terms in a heap
            if len(top_terms) < 75:
                heapq.heappush(top_terms, (chi_squared, term))
            else:
                heapq.heappushpop(top_terms, (chi_squared, term))

        # Create a string with the 75 terms with the highest chi squared value
        # sorted by chi squared value in descending order
        # in the format "term1:chi_squared1 term2:chi_squared2 ... term75:chi_squared75"
        terms = ["%s:%f" % (term, chi_squared) for (chi_squared, term) in heapq.nlargest(75, top_terms)]

        # emit the category enclosed in chevrons and the string of terms and chi squared values
        yield "<%s>" % key, " ".join(terms)

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
