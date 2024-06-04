"""Filter functions useful to filter data based on their content."""

import logging

import numexpr as ne
import pandas as pd

logger = logging.getLogger("datasets_utils._filters")


def filter_by_header(header: dict | None, filters: list | None) -> bool:
    """Filter the data based on the header keys.

    Parameters
    ----------
    header : dict
        Dictionary containing the header keys and values.
    filters : list
        Dictionary containing the filters to apply to the header.

    Returns
    -------
    bool
        True if the header passes the filters, False otherwise.

    Raises
    ------
    KeyError
        If a key in the filters is not found in the header.

    """
    if filters is None or header is None:
        logger.info("Either filters or header is None. Returning True.")
        return True

    for _filter in filters:
        try:
            test_passed = ne.evaluate(_filter, local_dict=header, global_dict={}).all()
            if not test_passed:
                logger.info("Filter `%s` returner False.", _filter)
                return False
        except KeyError as exc:
            msg = f"Key {exc} not found in header"
            logger.exception(msg)
            KeyError(msg)

    logger.info("Header passed all filters.")
    return True


def filter_by_queries(
    data: pd.DataFrame | None,
    filters: dict | None,
) -> pd.DataFrame | None:
    """Filter the dataframe using queries based on the columns.

    Parameters
    ----------
    data : pd.DataFrame | None
        Dataframe to filter.
    filters : dict | None
        Dictionary containing the filters to apply to the dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with the filtered data. An empty dataframe is returned if no
        data passed the filters.

    Raises
    ------
    TypeError
        If a filter query is not a string.

    """
    if filters is None or data is None:
        logger.info("Either filters or header is None. Returning the dataframe.")
        return data

    filtered_data = data.copy()
    for key, query in filters.items():
        if not isinstance(query, str):
            logger.error("Filter query must be a string.")
            raise TypeError(query)

        logger.info("Filtering data based on query: %s", key)
        filtered_data = filtered_data.query(query)

    if filtered_data.empty:
        logger.info("No data passed the filters. Dataframe is empty")
    else:
        logger.info(
            "Dataframe was filtered successfully. Lines filtered: %d",
            len(data) - len(filtered_data),
        )
    return filtered_data
