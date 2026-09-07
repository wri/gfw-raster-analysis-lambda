import numpy as np
import pandas as pd
import pytest

from raster_analysis.data_environment import DataEnvironment
from raster_analysis.exceptions import QueryParseException
from raster_analysis.query import (
    Function,
    Query,
    Selector,
    Sort,
    _parse_group_by,
    _parse_order_by,
)


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
        assert isinstance(
            sort, Sort
        ), f"Expected Sort enum, got {type(sort).__name__}: {sort!r}"

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


class TestDateFilterEncoding:
    """
    Regression tests for the production bug:

        ValueError: only leading negative signs are allowed

    Root cause
    ----------
    ``Selector.__hash__`` hashes only ``layer``, but Pydantic's default
    ``__eq__`` compares all fields including ``alias``. ``dict.fromkeys``
    removes a key as duplicate only when both its hash AND equality match a
    key already in the dict.

    Because the SELECT clause produces ``Selector(layer=..., alias='date')``
    and the GROUP BY clause produces ``Selector(layer=..., alias=None)``, the
    two objects have the same hash but are *not equal* under Pydantic's eq.
    ``dict.fromkeys`` therefore keeps both, and ``Query.get_result_selectors``
    returns a list with two entries for the same layer.

    ``AnalysisTiler._postprocess_results`` loops over that list and calls
    ``DataEnvironment.decode_layer`` once per selector. The first call
    correctly converts uint16 pixel offsets (e.g. ``4019``) to ISO date
    strings (``'2026-01-01'``). The second call then tries to decode those
    already-decoded strings using the expression::

        A.astype('timedelta64[D]') + datetime64('2015-01-01', 'D')

    Pandas calls ``pd.to_timedelta('2026-01-01')`` internally, and its
    timedelta parser rejects the ``-`` separators at non-leading positions:

        ValueError: only leading negative signs are allowed

    Fix
    ---
    Add ``Selector.__eq__`` so it matches ``Selector.__hash__`` — both use
    ``layer`` only. ``alias`` is a presentational concern (column renaming at
    the very end of postprocessing) and must not affect deduplication.
    """

    # These are the exact decode/encode expressions that gfw-data-api generates
    # in _get_date_conf_derived_layers() for every alert date layer.
    _DECODE_EXPRESSION = (
        "(A.astype('timedelta64[D]') + datetime64('2015-01-01', 'D')).astype(str)"
    )
    _ENCODE_EXPRESSION = (
        "(datetime64(A, 'D') - datetime64('2015-01-01', 'D')).astype(uint16)"
    )

    _ALERT_DATE_CONF_SOURCE = {
        "source_uri": "s3://bucket/{tile_id}.tif",
        "tile_scheme": "nw",
        "grid": "10/100000",
        "name": "gfw_integrated_alerts__date_conf",
    }

    _ALERT_DATE_DERIVED = {
        "source_layer": "gfw_integrated_alerts__date_conf",
        "name": "gfw_integrated_alerts__date",
        "calc": "A % 10000",
        "decode_expression": _DECODE_EXPRESSION,
        "encode_expression": _ENCODE_EXPRESSION,
    }

    # The problematic query from production.
    _PRODUCTION_QUERY = (
        "SELECT gfw_integrated_alerts__date AS date, COUNT(*) AS alert_count "
        "FROM gfw_integrated_alerts__date "
        "WHERE gfw_integrated_alerts__date >= '2026-01-01' "
        "GROUP BY gfw_integrated_alerts__date "
        "ORDER BY gfw_integrated_alerts__date"
    )

    def _make_env(self):
        return DataEnvironment(
            layers=[self._ALERT_DATE_CONF_SOURCE, self._ALERT_DATE_DERIVED]
        )

    # ------------------------------------------------------------------
    # Bug-demonstrating tests (these FAIL before the fix, PASS after)
    # ------------------------------------------------------------------

    def test_selector_eq_is_inconsistent_with_hash(self):
        """
        Selector.__hash__ uses only layer, but Pydantic's default __eq__
        uses all fields. Two selectors that differ only in alias therefore
        have the same hash but compare as unequal.

        This inconsistency means dict.fromkeys cannot deduplicate them, so
        get_result_selectors() returns both the SELECT selector (alias='date')
        and the GROUP BY selector (alias=None) for the same column, causing
        decode_layer() to be called twice and raising the production error.

        After the fix, __eq__ also uses only layer, so this assertion passes.
        """
        s_with_alias = Selector(layer="gfw_integrated_alerts__date", alias="date")
        s_no_alias = Selector(layer="gfw_integrated_alerts__date", alias=None)

        assert hash(s_with_alias) == hash(
            s_no_alias
        ), "Selectors for the same layer must have the same hash"
        assert s_with_alias == s_no_alias, (
            "Selectors for the same layer must compare as equal so that "
            "dict.fromkeys deduplicates them. alias is presentational only."
        )

    def test_get_result_selectors_deduplicates_select_alias_and_group_by(self):
        """
        When the same column appears in SELECT (with AS alias) and GROUP BY
        (without alias), get_result_selectors() must return exactly one
        Selector — not two.

        Returning two causes decode_layer() to be called twice on the same
        DataFrame column, producing:
            ValueError: only leading negative signs are allowed
        """
        env = self._make_env()
        q = Query(self._PRODUCTION_QUERY, env)

        result_selectors = q.get_result_selectors()

        assert len(result_selectors) == 1, (
            f"Expected 1 selector for the date column, got {len(result_selectors)}: "
            + str([(s.layer, s.alias) for s in result_selectors])
        )
        assert result_selectors[0].layer == "gfw_integrated_alerts__date"
        # The alias from SELECT must be preserved for later column renaming.
        assert (
            result_selectors[0].alias == "date"
        ), "The retained selector should be the one with the alias (from SELECT)"

    # ------------------------------------------------------------------
    # Direct reproduction of the production error
    # ------------------------------------------------------------------

    def test_double_decode_raises_the_production_valueerror(self):
        """
        Calling decode_layer() twice on the same date column — exactly what
        the bug causes — reproduces the production error.

        First decode: uint16 pixel offsets → ISO date strings (correct).
        Second decode: date strings → pd.to_timedelta('2026-01-01') → raises
            ValueError: only leading negative signs are allowed
        """
        env = self._make_env()

        pixel_series = pd.Series([4019, 4020], dtype=np.uint16)
        date_strings = env.decode_layer("gfw_integrated_alerts__date", pixel_series)
        assert date_strings.tolist() == ["2026-01-02", "2026-01-03"]

        with pytest.raises(ValueError, match="only leading negative signs are allowed"):
            env.decode_layer("gfw_integrated_alerts__date", date_strings)

    # ------------------------------------------------------------------
    # Fix-verification tests (should always pass)
    # ------------------------------------------------------------------

    def test_encode_converts_date_string_to_uint16_pixel_offset(self):
        """
        encode_layer must convert an ISO date string to the uint16 pixel
        offset stored in the raster (days since 2015-01-01).
        2026-01-01 is 4019 days after 2015-01-01 → uint16(4019).
        """
        env = self._make_env()
        encoded = env.encode_layer("gfw_integrated_alerts__date", "2026-01-01")

        assert len(encoded) == 1
        assert isinstance(
            encoded[0], np.uint16
        ), f"Expected numpy.uint16, got {type(encoded[0]).__name__}"
        assert encoded[0] == np.uint16(4018)

    def test_decode_is_inverse_of_encode(self):
        """
        Encoding a date to a pixel value and decoding it back must round-trip
        to the original date string.
        """
        env = self._make_env()
        for date_str in ["2026-01-01", "2025-06-15", "2024-12-31"]:
            encoded = env.encode_layer("gfw_integrated_alerts__date", date_str)
            decoded = env.decode_layer(
                "gfw_integrated_alerts__date",
                pd.Series(encoded, dtype=np.uint16),
            )
            assert decoded.iloc[0] == date_str, (
                f"Round-trip failed for '{date_str}': "
                f"encoded to {encoded[0]}, decoded to '{decoded.iloc[0]}'"
            )

    def test_filter_apply_produces_correct_mask(self):
        """
        The full filter chain (parse -> encode -> apply) must produce a
        correct boolean mask using the production encode_expression.

        Pixel values:
            3900 ~ 2025-09-05  (before 2026-01-01) -> excluded
            4019 = 2026-01-01  (boundary)           -> included
            4020 = 2026-01-02  (after boundary)     -> included
            4000 ~ 2025-11-06  (before 2026-01-01)  -> excluded
        """
        from raster_analysis.query import _parse_filter

        env = self._make_env()
        where_clause = {
            "gte": ["gfw_integrated_alerts__date", {"literal": "2026-01-01"}]
        }
        raster = np.array([[3900, 4018], [4019, 4000]], dtype=np.uint16)

        class _MockWindow:
            def __init__(self, data):
                self.data = data

        filter_node = _parse_filter(where_clause, env)
        mask = filter_node.apply(
            tile_width=2,
            windows={"gfw_integrated_alerts__date": _MockWindow(raster)},
        )

        assert mask.shape == raster.shape
        assert bool(mask[0, 0]) is False, "3900 (~2025-09-05) should be excluded"
        assert bool(mask[0, 1]) is True, "4018 (2026-01-01 boundary) should be included"
        assert bool(mask[1, 0]) is True, "4019 (2026-01-02) should be included"
        assert bool(mask[1, 1]) is False, "4000 (~2025-11-06) should be excluded"

    def test_no_double_decode_after_fix(self):
        """
        After the fix, iterating over get_result_selectors() and calling
        decode_layer() once per selector must succeed with no double-decode.

        This simulates the decode loop in AnalysisTiler._postprocess_results.
        """
        env = self._make_env()
        q = Query(self._PRODUCTION_QUERY, env)

        raw = pd.DataFrame(
            {
                "gfw_integrated_alerts__date": pd.Series(
                    [4019, 4020, 4019], dtype=np.int64
                ),
                "count": pd.Series([5, 3, 7], dtype=np.int64),
            }
        )

        for selector in q.get_result_selectors():
            raw[selector.layer] = env.decode_layer(selector.layer, raw[selector.layer])

        assert raw["gfw_integrated_alerts__date"].tolist() == [
            "2026-01-02",
            "2026-01-03",
            "2026-01-02",
        ]
