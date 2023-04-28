import json
import logging

from mrjob.job import MRJob

logger = logging.getLogger(__name__)


class CategoryCounter(MRJob):
    """
    This job counts the number of reviews in each category.
    """

    def jobconf(self):
        """
        Set the number of reducers.
        :return: A dictionary of job configuration options.
        """
        # set the number of reducers using the jobconf dictionary
        orig_jobconf = super().jobconf()
        jobconf = {'mapreduce.job.reduces': 2}
        jobconf.update(orig_jobconf)
        return jobconf

    def mapper(self, _, line):
        """
        Map each review to a category.
        :param _: A key.
        Unused.
        :param line: A single line of the input file.
        Represents a single review.
        :return: Yields the category and a count of 1.
        """
        # load json data from line
        review = json.loads(line)

        # extract the product category
        category = review["category"]

        # emit the category and the document count
        yield category, 1

    def combiner(self, key, values):
        """
        Combine the counts for each category.
        :param key: A category.
        :param values: A list of counts.
        :return: Yields the category and the sum of the counts.
        """
        yield key, sum(values)

    def reducer(self, key, values):
        """
        Combine the counts for each category.
        :param key: A category.
        :param values: A list of counts.
        :return: Yields the category and the sum of the counts.
        """
        yield key, sum(values)


if __name__ == "__main__":
    CategoryCounter.run()
