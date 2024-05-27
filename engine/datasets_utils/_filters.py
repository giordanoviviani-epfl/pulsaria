"""Filter functions useful to filter data based on their content."""

import logging
import operator

import pandas as pd

logger = logging.getLogger("datasets_utils._filters")


def filter_by_header(header: dict | None, filters: dict | None) -> bool:
    """Filter the data based on the header keys.

    Parameters
    ----------
    header : dict
        Dictionary containing the header keys and values.
    filters : dict
        Dictionary containing the filters to apply to the header.

    Returns
    -------
    bool
        True if the header passes the filters, False otherwise.

    Raises
    ------
    KeyError
        If a key in the filters is not found in the header.
    TypeError
        If one of the filters is not a dictionary.
    KeyError
        If one of the filters does not contain the keys "operator" and "value".

    """
    if filters is None or header is None:
        logger.info("Either filters or header is None. Returning True.")
        return True

    for key, header_filter in filters.items():
        if key not in header:
            logger.error("Key %s not found in header.", key)
            raise KeyError(key)

        if not isinstance(header_filter, dict):
            logger.error("Filters must be a dictionary.")
            raise TypeError(header_filter)

        if not {"operator", "value"}.issubset(set(header_filter)):
            logger.error("Filter must contain 'operator' and 'value' keys.")
            raise KeyError(header_filter)

        if isinstance(header_filter["operator"], str):
            header_filter["operator"] = [header_filter["operator"]]
            header_filter["value"] = [header_filter["value"]]

        for operator_str, value in zip(
            header_filter["operator"],
            header_filter["value"],
            strict=True,
        ):
            filter_operator = getattr(operator, operator_str)
            if not filter_operator(header[key], value):
                logger.info("Header key %s did not pass the filter.", key)
                return False

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
