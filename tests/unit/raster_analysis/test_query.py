import pytest

from raster_analysis.exceptions import QueryParseException
from raster_analysis.query import Function, _parse_group_by, _parse_order_by, Sort


class TestParseGroupBy:
    """
    Tests demonstrating fixes for several edge cases in _parse_group_by:

    1. KeyError: 'value' when the parsed structure lacks that key
    2. Silent drop of unrecognized functions (e.g. year(...))
    3. Silent drop of unexpected value types (e.g. int, list)
    """

    def test_missing_value_key_raises_informative_exception(self):
        """
        mo_sql_parsing may produce structures without a 'value' key in some edge
        cases (e.g. malformed queries or parser version differences). The old code
        raised a bare KeyError with the message 'value', which is exactly the
        cryptic one-word 500 error seen in Lambda logs.
        """
        PARSED_MISSING_VALUE_KEY = {
            "groupby": [{"name": "umd_tree_cover_loss__year"}],  # 'value' key absent
            "from": "umd_tree_cover_loss__year",
        }

        with pytest.raises(QueryParseException) as exc_info:
            _parse_group_by(PARSED_MISSING_VALUE_KEY)

        error_message = str(exc_info.value)
        assert "missing 'value' key" in error_message
        assert "name" in error_message  # shows the actual keys present

    def test_better_message_when_raising_on_unsupported_functions_(self):
        """
        `year(...)` is recognized by mo_sql_parsing but is not in the Function
        enum. The old code hit the dict branch and then raised QueryParseException.
        But a function like `date(...)` or `quarter(...)` would do the same.
        The real silent-drop risk is if func_name somehow doesn't match — see
        below for the subtler case.

        Still raise for unknown functions in GROUP BY but make the error message
        more informative.
        """
        PARSED_UNKNOWN_FUNC = {
            "groupby": [{"value": {"year": "umd_tree_cover_loss__year"}}],
            "from": "umd_tree_cover_loss__year",
        }

        with pytest.raises(QueryParseException) as exc_info:
            _parse_group_by(PARSED_UNKNOWN_FUNC)

        error_message = str(exc_info.value)
        assert "year" in error_message
        assert "umd_tree_cover_loss__year" in error_message
        for supported in Function.__members__.values():
            assert supported in error_message


    def test_raises_on_int_value(self):
        """
        If group["value"] is neither a dict nor a str — for example an int
        — the old code silently skipped it.
        """
        PARSED_INT_VALUE = {
            "groupby": [{"value": 2021}],
            "from": "umd_tree_cover_loss__year",
        }

        with pytest.raises(QueryParseException) as exc_info:
            _parse_group_by(PARSED_INT_VALUE)

        error_message = str(exc_info.value)
        assert "int" in error_message
        assert "2021" in error_message

    def test_raises_on_list_value(self):
        """
        If group["value"] is neither a dict nor a str — for example a
        list — the old code silently skipped it.
        """
        PARSED_LIST_VALUE = {
            "groupby": [{"value": ["umd_tree_cover_loss__year", "something_else"]}],
            "from": "umd_tree_cover_loss__year",
        }
        with pytest.raises(QueryParseException) as exc_info:
            _parse_group_by(PARSED_LIST_VALUE)

        error_message = str(exc_info.value)
        assert "list" in error_message


    # ---------------------------------------------------------------------------
    # Sanity checks: valid inputs still work correctly
    # ---------------------------------------------------------------------------

    def test_handle_string_group_by(self):
        parsed = {
            "groupby": [{"value": "umd_tree_cover_loss__year"}],
            "from": "umd_tree_cover_loss__year",
        }
        result = _parse_group_by(parsed)

        assert len(result) == 1
        assert result[0].layer == "umd_tree_cover_loss__year"

    def test_handle_known_function_group_by(self):
        # isoweek is the one entry in the Function enum
        parsed = {
            "groupby": [{"value": {"isoweek": "umd_tree_cover_loss__year"}}],
            "from": "umd_tree_cover_loss__year",
        }
        result = _parse_group_by(parsed)

        assert len(result) == 1
        assert result[0].function == Function.isoweek

    def test_handle_no_group_by(self):
        parsed = {"from": "umd_tree_cover_loss__year"}
        assert _parse_group_by(parsed) == []

    def test_handle_multiple_group_by_columns(self):
        parsed = {
            "groupby": [
                {"value": "umd_tree_cover_loss__year"},
                {"value": "umd_tree_cover_density_2000__threshold"},
            ],
            "from": "umd_tree_cover_loss__year",
        }

        assert len(_parse_group_by(parsed)) == 2


class TestParseOrderBy:
    """
    Tests demonstrating bugs in the original _parse_order_by:

    1. KeyError: 'value' when the parsed structure lacks that key
    2. Inconsistent return type for sort — default is Sort.asc (enum) but
       parsed values are stored as raw str, causing potential comparison bugs
    3. Silent last-write-wins when multiple ORDER BY columns have conflicting
       sort directions
    """

    def test_missing_value_key_raises_informative_exception(self):
        """
        If the parsed structure lacks a 'value' key, the old code raises a
        bare KeyError('value') — the same cryptic one-word error seen in
        Lambda logs. The fixed code should raise a QueryParseException with
        a descriptive message instead.
        """
        parsed = {
            "orderby": [{"name": "umd_tree_cover_loss__year"}],  # 'value' key absent
            "from": "umd_tree_cover_loss__year",
        }

        with pytest.raises(QueryParseException) as exc_info:
            _parse_order_by(parsed)

        error_message = str(exc_info.value)
        assert "missing 'value' key" in error_message
        assert "name" in error_message  # shows the actual keys present

    def test_default_sort_is_sort_enum(self):
        """
        When no ORDER BY is present, sort defaults to Sort.asc (a Sort enum
        member). Downstream code relying on this will compare against Sort
        enum values.
        """
        parsed = {"from": "umd_tree_cover_loss__year"}
        _, sort = _parse_order_by(parsed)

        assert sort == Sort.asc
        assert isinstance(sort, Sort)

    def test_parsed_sort_should_also_be_sort_enum(self):
        """
        When a sort direction is parsed from the query, the old code stores
        it as a plain str via order_by["sort"].lower(), not a Sort enum value.
        This means the return type is inconsistent: Sort.asc by default but
        str when parsed, so `sort == Sort.asc` would be False even for an
        ascending query that explicitly specifies ASC.

        The fixed code should always return a Sort enum member.
        """
        parsed = {
            "orderby": [{"value": "umd_tree_cover_loss__year", "sort": "asc"}],
            "from": "umd_tree_cover_loss__year",
        }
        _, sort = _parse_order_by(parsed)

        assert sort == Sort.asc
        assert isinstance(sort, Sort), (
            f"Expected Sort enum, got {type(sort).__name__}: {sort!r}"
        )

    def test_parsed_desc_sort_is_sort_enum(self):
        parsed = {
            "orderby": [{"value": "umd_tree_cover_loss__year", "sort": "desc"}],
            "from": "umd_tree_cover_loss__year",
        }
        _, sort = _parse_order_by(parsed)

        assert sort == Sort.desc
        assert isinstance(sort, Sort)

    def test_conflicting_sort_directions_last_wins_silently(self):
        """
        With multiple ORDER BY columns with different sort directions, the old
        code overwrites sort on each iteration so the last column's direction
        wins silently. There is no error or warning that earlier directions
        were discarded.

        The fixed code should raise a QueryParseException when mixed sort
        directions are provided, since there is no unambiguous way to handle
        this with the current single sort return value.
        """
        parsed = {
            "orderby": [
                {"value": "umd_tree_cover_loss__year", "sort": "asc"},
                {"value": "umd_tree_cover_density_2000__threshold", "sort": "desc"},
            ],
            "from": "umd_tree_cover_loss__year",
        }

        with pytest.raises(QueryParseException) as exc_info:
            _parse_order_by(parsed)

        assert "sort" in str(exc_info.value).lower()

    # -----------------------------------------------------------------------
    # Sanity checks: valid inputs still work correctly
    # -----------------------------------------------------------------------

    def test_no_order_by(self):
        parsed = {"from": "umd_tree_cover_loss__year"}
        order_bys, sort = _parse_order_by(parsed)

        assert order_bys == []
        assert sort == Sort.asc

    def test_single_column_no_explicit_sort(self):
        parsed = {
            "orderby": [{"value": "umd_tree_cover_loss__year"}],
            "from": "umd_tree_cover_loss__year",
        }
        order_bys, sort = _parse_order_by(parsed)

        assert len(order_bys) == 1
        assert order_bys[0].layer == "umd_tree_cover_loss__year"
        assert sort == Sort.asc

    def test_single_column_explicit_asc(self):
        parsed = {
            "orderby": [{"value": "umd_tree_cover_loss__year", "sort": "asc"}],
            "from": "umd_tree_cover_loss__year",
        }
        order_bys, sort = _parse_order_by(parsed)

        assert len(order_bys) == 1
        assert sort == Sort.asc

    def test_single_column_explicit_desc(self):
        parsed = {
            "orderby": [{"value": "umd_tree_cover_loss__year", "sort": "desc"}],
            "from": "umd_tree_cover_loss__year",
        }
        order_bys, sort = _parse_order_by(parsed)

        assert len(order_bys) == 1
        assert sort == Sort.desc

    def test_multiple_columns_same_sort_direction(self):
        parsed = {
            "orderby": [
                {"value": "umd_tree_cover_loss__year", "sort": "desc"},
                {"value": "umd_tree_cover_density_2000__threshold", "sort": "desc"},
            ],
            "from": "umd_tree_cover_loss__year",
        }
        order_bys, sort = _parse_order_by(parsed)

        assert len(order_bys) == 2
        assert sort == Sort.desc
