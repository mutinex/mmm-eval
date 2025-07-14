from typing import Any

from google.cloud import bigquery
from google.cloud.bigquery.table import RowIterator, _EmptyRowIterator


class BigQueryLoader:
    def __init__(self, project) -> None:
        """Initialize a BigQuery loader with a project.

        Parameters
        ----------
        project : str
            The name of the project to load data from.

        """
        self.client = bigquery.Client(project=project)

    def load_data(self, query) -> Any:
        """Load data from BigQuery using a SQL query.

        Parameters
        ----------
        query : str
            The SQL query to execute.

        Returns
        -------
        df : pandas.DataFrame
            The results of the query as a pandas DataFrame.

        """
        query_job: bigquery.QueryJob = self.client.query(query)
        results: RowIterator | _EmptyRowIterator = query_job.result()
        return results.to_dataframe()