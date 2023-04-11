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
        self.category_count_encoding = 'cc'
        self.a_encoding = 'A'
        self.b_encoding = 'B'
        self.c_encoding = 'C'
        self.n_encoding = 'N'

    def set_up_logging(cls, quiet=False, verbose=False, stream=None):
        log_to_stream(name="mrjob", debug=verbose, stream=stream)
        log_to_stream(name="__main__", debug=verbose, stream=stream)
        log_to_stream(name="amazon_reviews_chi_squared", debug=verbose, stream=stream)

    def mapper_init(self):
        self.stopwords = set()
        with open("stopwords.txt", "r") as f:
            for line in f.read().splitlines():
                self.stopwords.add(line.strip())
        logger.debug("Stopwords loaded: %d" % len(self.stopwords))

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

        # emit the category and a count of 1
        logger.debug("category: %s, count: %d" % (category, 1))
        yield category, (self.category_count_encoding, 1)

        # emit None the category and a count of 1
        logger.debug("category: %s, count: %d" % (category, 1))
        yield None, (category, 1)

        # emit the category and the term
        for term in terms:
            logger.debug("category: %s, term: %s" % (category, term))
            yield category, (self.term_encoding, term)
            logger.debug("term: %s, category: %s" % (term, category))
            yield term, (self.category_encoding, category)

    def reducer_calculate_variables(self, key, values):
        if key is None:
            number_of_reviews = 0
            distinct_categories = set()
            for (category, value) in values:
                distinct_categories.add(category)
                number_of_reviews += value

            for category in distinct_categories:
                logger.debug("category: %s, number of reviews: %d" % (category, number_of_reviews))
                yield category, (self.n_encoding, None, number_of_reviews)
        else:
            terms_in_category = []
            categories_for_term = []
            category_count = 0

            for (encoding, value) in values:
                if encoding == self.term_encoding:
                    terms_in_category.append(value)
                elif encoding == self.category_encoding:
                    categories_for_term.append(value)
                elif encoding == self.category_count_encoding:
                    category_count += value

            if len(terms_in_category) > 0:
                # count the number of occurrences of each term in the category
                term_counts = Counter(terms_in_category)
                for term in term_counts.keys():
                    a = term_counts[term]
                    c = category_count - a
                    yield key, (self.a_encoding, term, a)
                    yield key, (self.c_encoding, term, c)

            if len(categories_for_term) > 0:
                # count the number of occurrences of each category for the term
                category_counts = Counter(categories_for_term)
                for category in category_counts.keys():
                    b = sum(category_counts.values()) - category_counts[category]
                    yield category, (self.b_encoding, key, b)

    def reducer_chi_squared(self, key, values):
        # create a dictionary for each term
        term_dict = {}

        # extract A, B, C, and D values for each term
        # and the total number of reviews
        n = 0
        for (encoding, term, value) in values:
            if encoding == self.n_encoding:
                n = value
            else:
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
            d = n - a - b - c
            logger.debug("term: %s, a: %d, b: %d, c: %d, d: %d, n: %d" % (term, a, b, c, d, n))

            numerator = n * ((a * d - b * c) ** 2)
            denominator = (a + c) * (b + d) * (a + b) * (c + d)
            term_dict[term]["chi_squared"] = numerator / denominator

        # Create a string with the 75 terms with the highest chi squared value sorted by chi squared value in
        # descending order in the format "term1:chi_squared1 term2:chi_squared2 ... term75:chi_squared75"
        terms = sorted(term_dict.keys(), key=lambda x: term_dict[x]["chi_squared"], reverse=True)
        terms = terms[:75]
        terms = ["%s:%f" % (term, term_dict[term]["chi_squared"]) for term in terms]

        # emit the category and the string
        yield key, " ".join(terms)

    def steps(self):
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                reducer=self.reducer_calculate_variables,
            ),
            MRStep(
                reducer=self.reducer_chi_squared,
            )
        ]


if __name__ == '__main__':
    AmazonReviewsChiSquared.run()
