using System.Globalization;

namespace OnnxTextEmbeddings;

public enum SearchComparisonOperator
{
    Equal = 1,
    NotEqual = 2,
    GreaterThan = 3,
    GreaterThanOrEqual = 4,
    LessThan = 5,
    LessThanOrEqual = 6
}

public enum SearchSetOperator
{
    In = 1,
    NotIn = 2
}

public enum SearchNullOperator
{
    IsNull = 1,
    IsNotNull = 2
}

public enum SearchStringOperator
{
    Contains = 1,
    StartsWith = 2,
    EndsWith = 3
}

public enum SearchLogicalOperator
{
    And = 1,
    Or = 2
}

/// <summary>
/// Portable pre/post-filter expression understood by the in-memory engine and translated into parameterized SQL by
/// database providers. Provider-specific escape hatches remain available for predicates outside this common subset.
/// </summary>
public abstract record SearchFilter
{
    public static SearchFilter Equal(string field, object? value) => new SearchComparisonFilter(field, SearchComparisonOperator.Equal, value);
    public static SearchFilter NotEqual(string field, object? value) => new SearchComparisonFilter(field, SearchComparisonOperator.NotEqual, value);
    public static SearchFilter GreaterThan(string field, object value) => new SearchComparisonFilter(field, SearchComparisonOperator.GreaterThan, value);
    public static SearchFilter GreaterThanOrEqual(string field, object value) => new SearchComparisonFilter(field, SearchComparisonOperator.GreaterThanOrEqual, value);
    public static SearchFilter LessThan(string field, object value) => new SearchComparisonFilter(field, SearchComparisonOperator.LessThan, value);
    public static SearchFilter LessThanOrEqual(string field, object value) => new SearchComparisonFilter(field, SearchComparisonOperator.LessThanOrEqual, value);
    public static SearchFilter In<T>(string field, IEnumerable<T> values) => new SearchSetFilter(field, SearchSetOperator.In, values.Cast<object?>().ToArray());
    public static SearchFilter NotIn<T>(string field, IEnumerable<T> values) => new SearchSetFilter(field, SearchSetOperator.NotIn, values.Cast<object?>().ToArray());
    public static SearchFilter IsNull(string field) => new SearchNullFilter(field, SearchNullOperator.IsNull);
    public static SearchFilter IsNotNull(string field) => new SearchNullFilter(field, SearchNullOperator.IsNotNull);
    public static SearchFilter Contains(string field, string value) => new SearchStringFilter(field, SearchStringOperator.Contains, value);
    public static SearchFilter StartsWith(string field, string value) => new SearchStringFilter(field, SearchStringOperator.StartsWith, value);
    public static SearchFilter EndsWith(string field, string value) => new SearchStringFilter(field, SearchStringOperator.EndsWith, value);

    public static SearchFilter And(params SearchFilter[] filters) => Logical(SearchLogicalOperator.And, filters);
    public static SearchFilter Or(params SearchFilter[] filters) => Logical(SearchLogicalOperator.Or, filters);
    public static SearchFilter Not(SearchFilter filter) => new SearchNotFilter(filter ?? throw new ArgumentNullException(nameof(filter)));

    public static SearchFilter? CombineAnd(SearchFilter? left, SearchFilter? right)
    {
        if (left is null) return right;
        if (right is null) return left;
        return And(left, right);
    }

    private static SearchFilter Logical(SearchLogicalOperator @operator, IReadOnlyList<SearchFilter> filters)
    {
        ArgumentNullException.ThrowIfNull(filters);
        if (filters.Count == 0)
            throw new ArgumentException("At least one filter is required.", nameof(filters));
        if (filters.Any(filter => filter is null))
            throw new ArgumentException("Filters cannot contain null values.", nameof(filters));
        return filters.Count == 1 ? filters[0] : new SearchLogicalFilter(@operator, filters.ToArray());
    }
}

public sealed record SearchComparisonFilter(string Field, SearchComparisonOperator Operator, object? Value) : SearchFilter;
public sealed record SearchSetFilter(string Field, SearchSetOperator Operator, IReadOnlyList<object?> Values) : SearchFilter;
public sealed record SearchNullFilter(string Field, SearchNullOperator Operator) : SearchFilter;
public sealed record SearchStringFilter(string Field, SearchStringOperator Operator, string Value) : SearchFilter;
public sealed record SearchLogicalFilter(SearchLogicalOperator Operator, IReadOnlyList<SearchFilter> Filters) : SearchFilter;
public sealed record SearchNotFilter(SearchFilter Inner) : SearchFilter;

public static class SearchFilterEvaluator
{
    public static bool Matches(SearchFilter? filter, IReadOnlyDictionary<string, object?> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return filter is null || Evaluate(filter, values);
    }

    private static bool Evaluate(SearchFilter filter, IReadOnlyDictionary<string, object?> values) => filter switch
    {
        SearchComparisonFilter comparison => Compare(comparison, Resolve(values, comparison.Field)),
        SearchSetFilter set => Set(set, Resolve(values, set.Field)),
        SearchNullFilter nullFilter => nullFilter.Operator == SearchNullOperator.IsNull
            ? Resolve(values, nullFilter.Field) is null
            : Resolve(values, nullFilter.Field) is not null,
        SearchStringFilter text => Text(text, Resolve(values, text.Field)),
        SearchLogicalFilter logical => logical.Operator == SearchLogicalOperator.And
            ? logical.Filters.All(item => Evaluate(item, values))
            : logical.Filters.Any(item => Evaluate(item, values)),
        SearchNotFilter not => !Evaluate(not.Inner, values),
        _ => throw new NotSupportedException($"Unsupported search filter type '{filter.GetType().Name}'.")
    };

    private static object? Resolve(IReadOnlyDictionary<string, object?> values, string field)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field);
        if (!values.TryGetValue(field, out var value))
            throw new ArgumentException($"Search document does not define filter field '{field}'.", nameof(values));
        return value;
    }

    private static bool Compare(SearchComparisonFilter filter, object? actual)
    {
        if (filter.Operator == SearchComparisonOperator.Equal)
            return Equal(actual, filter.Value);
        if (filter.Operator == SearchComparisonOperator.NotEqual)
            return !Equal(actual, filter.Value);
        if (actual is null || filter.Value is null)
            return false;

        var comparison = CompareValues(actual, filter.Value);
        return filter.Operator switch
        {
            SearchComparisonOperator.GreaterThan => comparison > 0,
            SearchComparisonOperator.GreaterThanOrEqual => comparison >= 0,
            SearchComparisonOperator.LessThan => comparison < 0,
            SearchComparisonOperator.LessThanOrEqual => comparison <= 0,
            _ => throw new ArgumentOutOfRangeException(nameof(filter.Operator))
        };
    }

    private static bool Set(SearchSetFilter filter, object? actual)
    {
        var contains = filter.Values.Any(value => Equal(actual, value));
        return filter.Operator == SearchSetOperator.In ? contains : !contains;
    }

    private static bool Text(SearchStringFilter filter, object? actual)
    {
        if (actual is not string text)
            return false;
        return filter.Operator switch
        {
            SearchStringOperator.Contains => text.Contains(filter.Value, StringComparison.OrdinalIgnoreCase),
            SearchStringOperator.StartsWith => text.StartsWith(filter.Value, StringComparison.OrdinalIgnoreCase),
            SearchStringOperator.EndsWith => text.EndsWith(filter.Value, StringComparison.OrdinalIgnoreCase),
            _ => throw new ArgumentOutOfRangeException(nameof(filter.Operator))
        };
    }

    private static bool Equal(object? left, object? right)
    {
        if (left is string leftText && right is string rightText)
            return string.Equals(leftText, rightText, StringComparison.OrdinalIgnoreCase);
        return Equals(left, right);
    }

    private static int CompareValues(object left, object right)
    {
        if (left is string leftText && right is string rightText)
            return string.Compare(leftText, rightText, StringComparison.OrdinalIgnoreCase);

        if (left.GetType() != right.GetType() && left is IConvertible && right is IConvertible)
        {
            try { right = Convert.ChangeType(right, left.GetType(), CultureInfo.InvariantCulture)!; }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException) { }
        }

        if (left is IComparable comparable)
            return comparable.CompareTo(right);
        throw new ArgumentException($"Search filter value type '{left.GetType().Name}' is not comparable.");
    }
}

public sealed record SearchFilterSqlParameter(string Name, object? Value);
public sealed record CompiledSearchFilter(string Sql, IReadOnlyList<SearchFilterSqlParameter> Parameters);

/// <summary>
/// Database-neutral SQL predicate compiler. Providers supply the logical-field-to-quoted-column mapping; values are
/// always emitted as parameters. Atomic predicates are emitted as two-valued expressions so SQL NULL semantics match
/// the in-memory evaluator even when predicates are nested inside NOT. SQL-dialect-specific predicates can be layered
/// through each provider's existing native escape hatch.
/// </summary>
public static class SearchFilterSqlCompiler
{
    public static CompiledSearchFilter Compile(SearchFilter? filter, Func<string, string> resolveField)
    {
        ArgumentNullException.ThrowIfNull(resolveField);
        if (filter is null)
            return new CompiledSearchFilter(string.Empty, Array.Empty<SearchFilterSqlParameter>());

        var parameters = new List<SearchFilterSqlParameter>();
        var sql = CompileCore(filter, resolveField, parameters);
        return new CompiledSearchFilter(sql, parameters);
    }

    private static string CompileCore(
        SearchFilter filter,
        Func<string, string> resolveField,
        List<SearchFilterSqlParameter> parameters) => filter switch
    {
        SearchComparisonFilter comparison => CompileComparison(comparison, resolveField, parameters),
        SearchSetFilter set => CompileSet(set, resolveField, parameters),
        SearchNullFilter nullFilter => $"{resolveField(nullFilter.Field)} IS {(nullFilter.Operator == SearchNullOperator.IsNull ? "NULL" : "NOT NULL")}",
        SearchStringFilter text => CompileString(text, resolveField, parameters),
        SearchLogicalFilter logical => "(" + string.Join(
            logical.Operator == SearchLogicalOperator.And ? " AND " : " OR ",
            logical.Filters.Select(item => CompileCore(item, resolveField, parameters))) + ")",
        SearchNotFilter not => $"NOT ({CompileCore(not.Inner, resolveField, parameters)})",
        _ => throw new NotSupportedException($"Unsupported search filter type '{filter.GetType().Name}'.")
    };

    private static string CompileComparison(
        SearchComparisonFilter filter,
        Func<string, string> resolveField,
        List<SearchFilterSqlParameter> parameters)
    {
        var field = resolveField(filter.Field);
        if (filter.Value is null)
        {
            return filter.Operator switch
            {
                SearchComparisonOperator.Equal => $"{field} IS NULL",
                SearchComparisonOperator.NotEqual => $"{field} IS NOT NULL",
                _ => "1 = 0"
            };
        }

        var parameter = Add(parameters, filter.Value);
        return filter.Operator switch
        {
            SearchComparisonOperator.Equal => $"({field} IS NOT NULL AND {field} = @{parameter})",
            SearchComparisonOperator.NotEqual => $"({field} IS NULL OR {field} <> @{parameter})",
            SearchComparisonOperator.GreaterThan => $"({field} IS NOT NULL AND {field} > @{parameter})",
            SearchComparisonOperator.GreaterThanOrEqual => $"({field} IS NOT NULL AND {field} >= @{parameter})",
            SearchComparisonOperator.LessThan => $"({field} IS NOT NULL AND {field} < @{parameter})",
            SearchComparisonOperator.LessThanOrEqual => $"({field} IS NOT NULL AND {field} <= @{parameter})",
            _ => throw new ArgumentOutOfRangeException(nameof(filter.Operator))
        };
    }

    private static string CompileSet(
        SearchSetFilter filter,
        Func<string, string> resolveField,
        List<SearchFilterSqlParameter> parameters)
    {
        var field = resolveField(filter.Field);
        var nonNull = filter.Values.Where(value => value is not null).ToArray();
        var hasNull = nonNull.Length != filter.Values.Count;
        if (nonNull.Length == 0)
        {
            if (!hasNull)
                return filter.Operator == SearchSetOperator.In ? "1 = 0" : "1 = 1";
            return filter.Operator == SearchSetOperator.In ? $"{field} IS NULL" : $"{field} IS NOT NULL";
        }

        var placeholders = nonNull.Select(value => "@" + Add(parameters, value)).ToArray();
        var set = $"({string.Join(", ", placeholders)})";
        if (filter.Operator == SearchSetOperator.In)
            return hasNull
                ? $"({field} IS NULL OR ({field} IS NOT NULL AND {field} IN {set}))"
                : $"({field} IS NOT NULL AND {field} IN {set})";

        return hasNull
            ? $"({field} IS NOT NULL AND {field} NOT IN {set})"
            : $"({field} IS NULL OR {field} NOT IN {set})";
    }

    private static string CompileString(
        SearchStringFilter filter,
        Func<string, string> resolveField,
        List<SearchFilterSqlParameter> parameters)
    {
        var field = resolveField(filter.Field);
        var escaped = EscapeLike(filter.Value);
        var value = filter.Operator switch
        {
            SearchStringOperator.Contains => $"%{escaped}%",
            SearchStringOperator.StartsWith => $"{escaped}%",
            SearchStringOperator.EndsWith => $"%{escaped}",
            _ => throw new ArgumentOutOfRangeException(nameof(filter.Operator))
        };
        var parameter = Add(parameters, value);
        return $"({field} IS NOT NULL AND {field} LIKE @{parameter} ESCAPE '\\')";
    }

    private static string Add(List<SearchFilterSqlParameter> parameters, object? value)
    {
        var name = $"ote_filter_{parameters.Count}";
        parameters.Add(new SearchFilterSqlParameter(name, value));
        return name;
    }

    private static string EscapeLike(string value) => value
        .Replace("\\", "\\\\", StringComparison.Ordinal)
        .Replace("%", "\\%", StringComparison.Ordinal)
        .Replace("_", "\\_", StringComparison.Ordinal);
}
