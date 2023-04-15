import json
import logging
import re
from collections import defaultdict

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.util import log_to_stream

logger = logging.getLogger(__name__)


class AmazonReviewsChiSquared(MRJob):

    def __init__(self, args=None):
        super().__init__(args)
        self.stopwords = None
        self.category_encoding = 'c'
        self.category_count_encoding = 'e'
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
        with open("stopwords.txt", "r") as f:
            self.stopwords = set(f.read().splitlines())

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

        yield (category, self.category_encoding), (self.category_count_encoding, 1)

        # emit the category and the term
        for term in terms:
            # emit the term and the category
            yield (term, self.term_encoding), (category, 1)
            # emit the category and the terms
            yield (category, self.category_encoding), (term, 1)

    def combiner(self, key, values):
        # extract the encoding
        _, encoding = key

        if encoding == self.category_encoding:
            # count the number of occurrences of each term in the category
            term_count_for_category = defaultdict(int)
            for term, count in values:
                term_count_for_category[term] += count

            # emit the term count for the category
            for term, count in term_count_for_category.items():
                yield key, (term, count)

        elif encoding == self.term_encoding:
            # count the number of occurrences of each category for the term
            categories_count_for_term = defaultdict(int)
            for category, count in values:
                categories_count_for_term[category] += count

            # emit the categories count for the term
            for category, count in categories_count_for_term.items():
                yield key, (category, count)

    def reducer_calculate_variables(self, key, values):
        # extract the encoding
        _, encoding = key

        if encoding == self.category_encoding:
            category, _ = key
            term_counts = defaultdict(int)
            for term, count in values:
                term_counts[term] += count
            category_count = term_counts.pop(self.category_count_encoding)
            for term, count in term_counts.items():
                yield category, (self.a_encoding, term, count)
                yield category, (self.c_encoding, term, category_count - count)

        elif encoding == self.term_encoding:
            term, _ = key
            category_counts = defaultdict(int)
            for category, count in values:
                category_counts[category] += count
            aggregate_count = sum(category_counts.values())
            for category, count in category_counts.items():
                yield category, (self.b_encoding, term, aggregate_count - count)

    def reducer_chi_squared(self, key, values):
        # create a dictionary for each term
        term_dict = {}

        # extract A, B, C, and D values for each term
        for (encoding, term, value) in values:
            entry = term_dict.setdefault(term, {"a": 0, "b": 0, "c": 0})
            if encoding == self.a_encoding:
                entry["a"] = value
            elif encoding == self.b_encoding:
                entry["b"] = value
            elif encoding == self.c_encoding:
                entry["c"] = value

        # calculate chi squared for each term
        chi_squared_terms = {}
        for term, term_values in term_dict.items():
            a = term_values["a"]
            b = term_values["b"]
            c = term_values["c"]
            n = self.options.n
            d = n - a - b - c

            numerator = n * ((a * d - b * c) ** 2)
            denominator = (a + c) * (b + d) * (a + b) * (c + d)
            # save the chi squared value into a new dictionary
            chi_squared_terms[term] = numerator / denominator

        # Create a string with the 75 terms with the highest chi squared value sorted by chi squared value in
        # descending order in the format "term1:chi_squared1 term2:chi_squared2 ... term75:chi_squared75"
        terms = []
        for term, chi_squared in sorted(chi_squared_terms.items(), key=lambda x: x[1], reverse=True)[:75]:
            terms.append("%s:%f" % (term, chi_squared))

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
