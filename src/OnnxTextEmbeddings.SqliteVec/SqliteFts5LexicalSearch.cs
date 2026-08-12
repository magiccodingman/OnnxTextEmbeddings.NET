using System.Globalization;
using System.Text;
using Microsoft.Data.Sqlite;

namespace OnnxTextEmbeddings.SqliteVec;

public enum SqliteFts5QueryMode
{
    Plain = 1,
    NativeSyntax = 2
}

public sealed record SqliteFts5Field(string Name, string Column);

public sealed record SqliteFts5Query
{
    public required string Table { get; init; }
    public required string ItemKeyColumn { get; init; }
    /// <summary>All FTS5 table columns in declaration order. Required because bm25() weights are positional.</summary>
    public required IReadOnlyList<string> ColumnOrder { get; init; }
    public IReadOnlyList<SqliteFts5Field> Fields { get; init; } = Array.Empty<SqliteFts5Field>();
    public SqliteFts5QueryMode QueryMode { get; init; } = SqliteFts5QueryMode.Plain;
    public SearchFilter? Filter { get; init; }
    public IReadOnlyDictionary<string, string> FilterColumns { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal);
    public string? AdditionalWhereSql { get; init; }
}

/// <summary>SQLite FTS5 lexical retrieval using the engine's real bm25() function and native column weights.</summary>
public sealed class SqliteFts5LexicalSearch
{
    public async Task<DatabaseLexicalSearchResult<TKey>> SearchAsync<TKey>(
        SqliteConnection connection,
        string query,
        SqliteFts5Query lexicalQuery,
        IReadOnlyList<SearchFieldWeight>? fields = null,
        DatabaseLexicalSearchOptions? options = null,
        Action<SqliteCommand>? configureFilterParameters = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentNullException.ThrowIfNull(lexicalQuery);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The SqliteConnection must already be open.");
        if (lexicalQuery.ColumnOrder.Count == 0)
            throw new ArgumentException("SQLite FTS5 ColumnOrder cannot be empty.", nameof(lexicalQuery));
        options ??= new DatabaseLexicalSearchOptions();
        options.Validate();

        var selected = fields ?? Array.Empty<SearchFieldWeight>();
        var (weights, searchColumns) = ResolveFields(lexicalQuery, selected);
        var table = SqliteVecSemanticSearch.QuoteIdentifier(lexicalQuery.Table);
        var itemKey = SqliteVecSemanticSearch.QuoteIdentifier(lexicalQuery.ItemKeyColumn);
        var weightsSql = string.Join(", ", weights.Select(weight => weight.ToString("R", CultureInfo.InvariantCulture)));
        var bm25 = $"bm25({table}, {weightsSql})";
        var ftsQuery = lexicalQuery.QueryMode == SqliteFts5QueryMode.NativeSyntax
            ? query
            : BuildPlainQuery(query);
        if (searchColumns.Count > 0)
            ftsQuery = $"{{{string.Join(' ', searchColumns.Select(QuoteFtsIdentifier))}}} : ({ftsQuery})";

        var portableFilter = SearchFilterSqlCompiler.Compile(
            lexicalQuery.Filter,
            logical => SqliteVecSemanticSearch.QuoteIdentifier(ResolveFilterColumn(lexicalQuery.FilterColumns, logical)));
        var where = new List<string> { $"{table} MATCH $ote_lexical_query" };
        if (!string.IsNullOrWhiteSpace(portableFilter.Sql)) where.Add(portableFilter.Sql);
        if (!string.IsNullOrWhiteSpace(lexicalQuery.AdditionalWhereSql)) where.Add($"({lexicalQuery.AdditionalWhereSql})");

        var sql = $"""
            SELECT {itemKey}, -({bm25}) AS ote_score
            FROM {table}
            WHERE {string.Join(" AND ", where)}
            ORDER BY {bm25}
            LIMIT $ote_lexical_top
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.AddWithValue("$ote_lexical_query", ftsQuery);
        command.Parameters.AddWithValue("$ote_lexical_top", options.Top);
        foreach (var parameter in portableFilter.Parameters)
            command.Parameters.AddWithValue("$" + parameter.Name, parameter.Value ?? DBNull.Value);
        configureFilterParameters?.Invoke(command);

        var results = new List<LexicalSearchResult<TKey>>(options.Top);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            results.Add(new LexicalSearchResult<TKey>
            {
                Item = SqliteVecSemanticSearch.ReadKey<TKey>(reader.GetValue(0)),
                Score = Convert.ToSingle(reader.GetValue(1), CultureInfo.InvariantCulture),
                Fields = Array.Empty<LexicalFieldMatch>(),
                Scoring = new LexicalScoringInfo(LexicalScoringProfiles.SqliteFts5Bm25, 1)
                {
                    K1 = 1.2f,
                    B = 0.75f,
                    Provider = "SQLite/FTS5",
                    Mode = lexicalQuery.QueryMode.ToString()
                }
            });
        }

        return new DatabaseLexicalSearchResult<TKey>
        {
            Results = results,
            Retrieval = new LexicalCandidateRetrievalInfo
            {
                Provider = "SQLite/FTS5",
                Mode = $"BM25/{lexicalQuery.QueryMode}",
                RequestedCount = options.Top,
                ReturnedCount = results.Count
            }
        };
    }

    private static (float[] Weights, IReadOnlyList<string> SearchColumns) ResolveFields(
        SqliteFts5Query query,
        IReadOnlyList<SearchFieldWeight> selected)
    {
        var weights = new float[query.ColumnOrder.Count];
        var positions = query.ColumnOrder.Select((name, index) => new { name, index })
            .ToDictionary(item => item.name, item => item.index, StringComparer.Ordinal);
        var mappings = query.Fields.ToDictionary(field => field.Name, field => field.Column, StringComparer.Ordinal);
        var searchColumns = new List<string>();

        if (selected.Count == 0)
        {
            foreach (var mapping in query.Fields)
            {
                if (!positions.TryGetValue(mapping.Column, out var index))
                    throw new ArgumentException($"FTS5 field column '{mapping.Column}' is not present in ColumnOrder.", nameof(query));
                weights[index] = 1f;
                searchColumns.Add(mapping.Column);
            }
            if (query.Fields.Count == 0)
                Array.Fill(weights, 1f);
            return (weights, searchColumns);
        }

        foreach (var field in selected)
        {
            if (field.Weight < 0 || !float.IsFinite(field.Weight))
                throw new ArgumentOutOfRangeException(nameof(selected), $"Lexical field '{field.Name}' has an invalid weight.");
            if (!mappings.TryGetValue(field.Name, out var column))
                throw new ArgumentException($"No FTS5 column was configured for logical field '{field.Name}'.", nameof(selected));
            if (!positions.TryGetValue(column, out var index))
                throw new ArgumentException($"FTS5 field column '{column}' is not present in ColumnOrder.", nameof(query));
            weights[index] = field.Weight;
            if (field.Weight > 0) searchColumns.Add(column);
        }
        return (weights, searchColumns);
    }

    private static string BuildPlainQuery(string value)
    {
        var terms = new List<string>();
        var builder = new StringBuilder();
        foreach (var rune in value.EnumerateRunes())
        {
            if (Rune.IsLetterOrDigit(rune))
                builder.Append(rune.ToString());
            else if (builder.Length > 0)
            {
                terms.Add(builder.ToString());
                builder.Clear();
            }
        }
        if (builder.Length > 0) terms.Add(builder.ToString());
        if (terms.Count == 0)
            throw new ArgumentException("The lexical query does not contain searchable letters or digits.", nameof(value));
        return string.Join(" AND ", terms.Select(term => $"\"{term.Replace("\"", "\"\"", StringComparison.Ordinal)}\""));
    }

    private static string QuoteFtsIdentifier(string identifier) =>
        $"\"{identifier.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";

    private static string ResolveFilterColumn(IReadOnlyDictionary<string, string> columns, string logical)
    {
        if (!columns.TryGetValue(logical, out var physical) || string.IsNullOrWhiteSpace(physical))
            throw new ArgumentException($"No SQLite FTS5 filter-column mapping was configured for logical field '{logical}'.");
        return physical;
    }
}
