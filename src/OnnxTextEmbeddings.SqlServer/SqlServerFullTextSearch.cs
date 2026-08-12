using System.Globalization;
using Microsoft.Data.SqlClient;

namespace OnnxTextEmbeddings.SqlServer;

public enum SqlServerFullTextQueryMode
{
    FreeText = 1,
    Contains = 2
}

public sealed record SqlServerFullTextField(string Name, string Column);

public sealed record SqlServerLexicalQuery
{
    public required string Table { get; init; }
    public required string ItemKeyColumn { get; init; }
    /// <summary>The unique key column used by the SQL Server full-text index and returned as KEY by *TABLE functions.</summary>
    public required string FullTextKeyColumn { get; init; }
    public IReadOnlyList<SqlServerFullTextField> Fields { get; init; } = Array.Empty<SqlServerFullTextField>();
    public SqlServerFullTextQueryMode QueryMode { get; init; } = SqlServerFullTextQueryMode.FreeText;
    public SearchFilter? Filter { get; init; }
    public IReadOnlyDictionary<string, string> FilterColumns { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal);
    public string? AdditionalWhereSql { get; init; }
    public int FieldFusionRankConstant { get; init; } = 60;
}

/// <summary>
/// SQL Server/Azure SQL native full-text retrieval. Weighted logical fields are searched independently through
/// FREETEXTTABLE/CONTAINSTABLE and rank-fused so callers can weight title/category/body without pretending RANK is a
/// BM25 score or directly comparable across independent column searches.
/// </summary>
public sealed class SqlServerFullTextSearch
{
    public async Task<DatabaseLexicalSearchResult<TKey>> SearchAsync<TKey>(
        SqlConnection connection,
        string query,
        SqlServerLexicalQuery lexicalQuery,
        IReadOnlyList<SearchFieldWeight>? fields = null,
        DatabaseLexicalSearchOptions? options = null,
        Action<SqlCommand>? configureFilterParameters = null,
        SqlTransaction? transaction = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentNullException.ThrowIfNull(lexicalQuery);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The SqlConnection must already be open.");
        if (lexicalQuery.FieldFusionRankConstant < 0)
            throw new ArgumentOutOfRangeException(nameof(lexicalQuery.FieldFusionRankConstant));
        options ??= new DatabaseLexicalSearchOptions();
        options.Validate();

        var selected = ResolveFields(lexicalQuery.Fields, fields ?? Array.Empty<SearchFieldWeight>());
        var aggregates = new Dictionary<TKey, Aggregate<TKey>>();
        var firstSeen = 0;
        foreach (var field in selected.Where(field => field.Weight > 0))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var rows = await SearchFieldAsync<TKey>(
                connection,
                query,
                lexicalQuery,
                field,
                options.Top,
                configureFilterParameters,
                transaction,
                cancellationToken).ConfigureAwait(false);
            for (var index = 0; index < rows.Count; index++)
            {
                var row = rows[index];
                if (!aggregates.TryGetValue(row.Key, out var aggregate))
                {
                    aggregate = new Aggregate<TKey>(row.Key, firstSeen++);
                    aggregates.Add(row.Key, aggregate);
                }
                var rank = index + 1;
                aggregate.Score += field.Weight / (lexicalQuery.FieldFusionRankConstant + rank);
                aggregate.Fields.Add(new LexicalFieldMatch
                {
                    Name = field.Name,
                    Weight = field.Weight,
                    Score = row.Rank
                });
            }
        }

        var results = aggregates.Values
            .OrderByDescending(item => item.Score)
            .ThenBy(item => item.FirstSeen)
            .Take(options.Top)
            .Select(item => new LexicalSearchResult<TKey>
            {
                Item = item.Key,
                Score = item.Score,
                Fields = item.Fields.OrderByDescending(field => field.Weight).ToArray(),
                Scoring = new LexicalScoringInfo(LexicalScoringProfiles.SqlServerFullTextRank, 1)
                {
                    Provider = "SQL Server/Azure SQL Full-Text Search",
                    Mode = $"{lexicalQuery.QueryMode}/weighted-field-RRF"
                }
            })
            .ToArray();

        return new DatabaseLexicalSearchResult<TKey>
        {
            Results = results,
            Retrieval = new LexicalCandidateRetrievalInfo
            {
                Provider = "SQL Server/Azure SQL Full-Text Search",
                Mode = $"{lexicalQuery.QueryMode}/weighted-field-RRF",
                RequestedCount = options.Top,
                ReturnedCount = results.Length
            }
        };
    }

    private static async Task<IReadOnlyList<FieldRow<TKey>>> SearchFieldAsync<TKey>(
        SqlConnection connection,
        string query,
        SqlServerLexicalQuery lexicalQuery,
        SearchFieldWeight field,
        int top,
        Action<SqlCommand>? configureFilterParameters,
        SqlTransaction? transaction,
        CancellationToken cancellationToken)
        where TKey : notnull
    {
        var mapping = lexicalQuery.Fields.First(item => string.Equals(item.Name, field.Name, StringComparison.Ordinal));
        var table = SqlServerSemanticSearch.QuoteIdentifierPath(lexicalQuery.Table);
        var itemKey = SqlServerSemanticSearch.QuoteIdentifier(lexicalQuery.ItemKeyColumn);
        var fullTextKey = SqlServerSemanticSearch.QuoteIdentifier(lexicalQuery.FullTextKeyColumn);
        var column = SqlServerSemanticSearch.QuoteIdentifier(mapping.Column);
        var function = lexicalQuery.QueryMode == SqlServerFullTextQueryMode.FreeText ? "FREETEXTTABLE" : "CONTAINSTABLE";
        var portableFilter = SearchFilterSqlCompiler.Compile(
            lexicalQuery.Filter,
            logical => $"t.{SqlServerSemanticSearch.QuoteIdentifier(ResolveFilterColumn(lexicalQuery.FilterColumns, logical))}");
        var where = new List<string>();
        if (!string.IsNullOrWhiteSpace(portableFilter.Sql)) where.Add(portableFilter.Sql);
        if (!string.IsNullOrWhiteSpace(lexicalQuery.AdditionalWhereSql)) where.Add($"({lexicalQuery.AdditionalWhereSql})");
        var whereSql = where.Count == 0 ? string.Empty : "WHERE " + string.Join(" AND ", where);

        var sql = $"""
            SELECT t.{itemKey}, ft.[RANK]
            FROM {function}({table}, {column}, @ote_lexical_query, {top}) AS ft
            INNER JOIN {table} AS t ON t.{fullTextKey} = ft.[KEY]
            {whereSql}
            ORDER BY ft.[RANK] DESC
            """;
        await using var command = new SqlCommand(sql, connection, transaction);
        command.Parameters.AddWithValue("@ote_lexical_query", query);
        foreach (var parameter in portableFilter.Parameters)
            command.Parameters.AddWithValue("@" + parameter.Name, parameter.Value ?? DBNull.Value);
        configureFilterParameters?.Invoke(command);

        var rows = new List<FieldRow<TKey>>(top);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            rows.Add(new FieldRow<TKey>(
                SqlServerSemanticSearch.ReadKey<TKey>(reader.GetValue(0)),
                Convert.ToSingle(reader.GetValue(1), CultureInfo.InvariantCulture)));
        return rows;
    }

    private static IReadOnlyList<SearchFieldWeight> ResolveFields(
        IReadOnlyList<SqlServerFullTextField> mappings,
        IReadOnlyList<SearchFieldWeight> selected)
    {
        if (mappings.Count == 0)
            throw new ArgumentException("At least one SQL Server full-text field mapping is required.", nameof(mappings));
        if (selected.Count == 0)
            return mappings.Select(field => new SearchFieldWeight(field.Name, 1f)).ToArray();
        var known = mappings.Select(field => field.Name).ToHashSet(StringComparer.Ordinal);
        foreach (var field in selected)
        {
            if (field.Weight < 0 || !float.IsFinite(field.Weight))
                throw new ArgumentOutOfRangeException(nameof(selected), $"Lexical field '{field.Name}' has an invalid weight.");
            if (!known.Contains(field.Name))
                throw new ArgumentException($"No SQL Server full-text column was configured for logical field '{field.Name}'.", nameof(selected));
        }
        return selected;
    }

    private static string ResolveFilterColumn(IReadOnlyDictionary<string, string> columns, string logical)
    {
        if (!columns.TryGetValue(logical, out var physical) || string.IsNullOrWhiteSpace(physical))
            throw new ArgumentException($"No SQL Server lexical filter-column mapping was configured for logical field '{logical}'.");
        return physical;
    }

    private sealed record FieldRow<TKey>(TKey Key, float Rank) where TKey : notnull;

    private sealed class Aggregate<TKey>(TKey key, int firstSeen) where TKey : notnull
    {
        public TKey Key { get; } = key;
        public int FirstSeen { get; } = firstSeen;
        public float Score { get; set; }
        public List<LexicalFieldMatch> Fields { get; } = new();
    }
}
