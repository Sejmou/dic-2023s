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

        # emit the category and the document count
        yield (category, self.category_encoding), (self.document_count_for_category_encoding, 1)

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
            # extract the category
            category, _ = key

            # count the number of occurrences of each term in the category
            term_counts_for_category = defaultdict(int)
            for term, count in values:
                term_counts_for_category[term] += count

            # emit the number of documents in the category
            category_count = term_counts_for_category.pop(self.document_count_for_category_encoding)
            yield category, (self.document_count_for_category_encoding, _, category_count)

            # emit the term count for the category
            for term, count in term_counts_for_category.items():
                yield category, (self.a_encoding, term, count)

        elif encoding == self.term_encoding:
            # extract the term
            term, _ = key

            # count the number of occurrences of each category for the term
            category_counts_for_term = defaultdict(int)
            for category, count in values:
                category_counts_for_term[category] += count

            # emit the number of documents not in the category for each term
            aggregate_count = sum(category_counts_for_term.values())
            for category, count in category_counts_for_term.items():
                yield category, (self.b_encoding, term, aggregate_count - count)

    def reducer_chi_squared(self, key, values):
        # create a dictionary for each term
        term_dict = {}
        category_count = 0

        # extract A, B, values for each term and the number of documents in the category
        for (encoding, term, value) in values:
            # extract the number of documents in the category
            if encoding == self.document_count_for_category_encoding:
                category_count = value
                continue

            # extract the dictionary for the term or create a new one if it does not exist
            entry = term_dict.setdefault(term, {
                "a": 0,
                "b": 0,
            })
            if encoding == self.a_encoding:
                entry["a"] = value
            elif encoding == self.b_encoding:
                entry["b"] = value

        # calculate chi squared for each term
        for term, term_values in term_dict.items():
            a = term_values["a"]
            b = term_values["b"]
            c = category_count - a
            n = self.options.n
            d = n - a - b - c

            # save the chi squared value into the dictionary
            term_dict[term] = n * ((a * d - b * c) ** 2) / ((a + c) * (b + d) * (a + b) * (c + d))

        # Create a string with the 75 terms with the highest chi squared value
        # sorted by chi squared value in descending order
        # in the format "term1:chi_squared1 term2:chi_squared2 ... term75:chi_squared75"
        terms = []
        for term, chi_squared in sorted(term_dict.items(), key=lambda x: x[1], reverse=True)[:75]:
            terms.append("%s:%f" % (term, chi_squared))

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
