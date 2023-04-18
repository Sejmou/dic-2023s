import json
import logging

from mrjob.job import MRJob

logger = logging.getLogger(__name__)


class CategoryCounter(MRJob):
    def mapper(self, _, line):
        # load json data from line
        review = json.loads(line)

        # extract the product category
        category = review["category"]

        # emit the category and the document count
        yield category, 1

    def combiner(self, key, values):
        yield key, sum(values)

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == "__main__":
    CategoryCounter.run()
