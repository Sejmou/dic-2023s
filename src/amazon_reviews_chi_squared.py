import json
import logging
import re
from collections import Counter

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.util import log_to_stream

logger = logging.getLogger(__name__)


class AmazonReviewsChiSquared(MRJob):
    FILES = ["../data/stopwords.txt"]

    def __init__(self, args=None):
        super().__init__(args)
        self.stopwords = None
        self.category_encoding = 'c'
        self.term_encoding = 't'
        self.a_encoding = 'A'
        self.b_encoding = 'B'
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
        self.stopwords = set()
        with open("stopwords.txt", "r") as f:
            for line in f.read().splitlines():
                self.stopwords.add(line.strip())
        # logger.debug("Stopwords loaded: %d" % len(self.stopwords))

    def mapper(self, _, line):
        # load json data from line
        review = json.loads(line)

        # extract the review text and the product category
        text = review["reviewText"]
        category = review["category"]

        # tokenizes each text by using space, tabs, digits, and the characters ()[]{}.!?,;:+=-_"'`~#@&*%€$§\/ as delimiters
        terms = re.split("[^a-zA-Z<>^|]+", text)
        # case folding
        terms = [token.lower() for token in terms]
        # remove terms with less than 2 characters
        terms = [token for token in terms if len(token) > 1]
        # retain unique terms
        terms = set(terms)
        # remove stopwords
        terms = [token for token in terms if token not in self.stopwords]

        # emit the category and the terms
        yield (category, self.category_encoding), terms

        # emit the category and the term
        for term in terms:
            # logger.debug("term: %s, category: %s" % (term, category))
            yield (term, self.term_encoding), category

    def combiner(self, key, values):
        # extract the encoding
        _, encoding = key

        if encoding == self.category_encoding:
            # count the number of lists in values
            category_count = 0
            # count the number of occurrences of each term in the category
            term_count_for_category = Counter()
            for value in values:
                category_count += 1
                term_count_for_category += Counter(value)

            yield key, category_count
            yield key, term_count_for_category
        elif encoding == self.term_encoding:
            categories_count_for_term = Counter(values)
            yield key, categories_count_for_term

    def reducer_calculate_variables(self, key, values):
        # extract the encoding
        key, encoding = key

        if encoding == self.category_encoding:
            term_counts = Counter()
            category_count = 0
            for value in values:
                if isinstance(value, int):
                    # logger.debug("value (int): %d" % value)
                    category_count += value
                else:
                    # logger.debug("value (other): %s" % value)
                    term_counts += value
            for term in term_counts.keys():
                a = term_counts[term]
                c = category_count - a
                yield key, (self.a_encoding, term, a)
                yield key, (self.c_encoding, term, c)

        elif encoding == self.term_encoding:
            category_counts = Counter()
            for value in values:
                category_counts += value
            for category in category_counts.keys():
                b = sum(category_counts.values()) - category_counts[category]
                yield category, (self.b_encoding, key, b)

    def reducer_chi_squared(self, key, values):
        # create a dictionary for each term
        term_dict = {}

        # extract A, B, C, and D values for each term
        for (encoding, term, value) in values:
            term_dict.setdefault(term, {"a": 0, "b": 0, "c": 0, "d": 0})
            if encoding == self.a_encoding:
                term_dict[term]["a"] = value
            elif encoding == self.b_encoding:
                term_dict[term]["b"] = value
            elif encoding == self.c_encoding:
                term_dict[term]["c"] = value

        # calculate chi squared for each term
        for term in term_dict.keys():
            a = term_dict[term]["a"]
            b = term_dict[term]["b"]
            c = term_dict[term]["c"]
            n = self.options.n
            d = n - a - b - c
            # logger.debug("term: %s, a: %d, b: %d, c: %d, d: %d, n: %d" % (term, a, b, c, d, n))

            numerator = n * ((a * d - b * c) ** 2)
            denominator = (a + c) * (b + d) * (a + b) * (c + d)
            term_dict[term]["chi_squared"] = numerator / denominator

        # Create a string with the 75 terms with the highest chi squared value sorted by chi squared value in
        # descending order in the format "term1:chi_squared1 term2:chi_squared2 ... term75:chi_squared75"
        terms = sorted(term_dict.keys(), key=lambda x: term_dict[x]["chi_squared"], reverse=True)
        terms = terms[:75]
        terms = ["%s:%f" % (term, term_dict[term]["chi_squared"]) for term in terms]

        # emit the category and the string
        # enclose the key in chevrons
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
