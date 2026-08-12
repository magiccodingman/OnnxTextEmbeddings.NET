using System.Globalization;
using Npgsql;

namespace OnnxTextEmbeddings.PgVector;

public enum PgTextSearchWeight
{
    D = 0,
    C = 1,
    B = 2,
    A = 3
}

public enum PgTextSearchQueryMode
{
    WebSearch = 1,
    Plain = 2,
    Phrase = 3,
    Native = 4
}

public enum PgTextSearchRankMode
{
    CoverDensity = 1,
    Frequency = 2
}

public sealed record PgTextSearchField(string Name, PgTextSearchWeight Weight);

public sealed record PgVectorLexicalQuery
{
    public required string Table { get; init; }
    public required string ItemKeyColumn { get; init; }
    public required string SearchVectorColumn { get; init; }
    public IReadOnlyList<PgTextSearchField> Fields { get; init; } = Array.Empty<PgTextSearchField>();
    public string TextSearchConfiguration { get; init; } = "english";
    public PgTextSearchQueryMode QueryMode { get; init; } = PgTextSearchQueryMode.WebSearch;
    public PgTextSearchRankMode RankMode { get; init; } = PgTextSearchRankMode.CoverDensity;
    public int Normalization { get; init; }
    public SearchFilter? Filter { get; init; }
    public IReadOnlyDictionary<string, string> FilterColumns { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal);
    public string? AdditionalWhereSql { get; init; }
}

/// <summary>PostgreSQL-native full-text retrieval using tsvector/tsquery and ts_rank/ts_rank_cd.</summary>
public sealed class PgVectorLexicalSearch
{
    public async Task<DatabaseLexicalSearchResult<TKey>> SearchAsync<TKey>(
        NpgsqlConnection connection,
        string query,
        PgVectorLexicalQuery lexicalQuery,
        IReadOnlyList<SearchFieldWeight>? fields = null,
        DatabaseLexicalSearchOptions? options = null,
        Action<NpgsqlCommand>? configureFilterParameters = null,
        NpgsqlTransaction? transaction = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentNullException.ThrowIfNull(lexicalQuery);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The NpgsqlConnection must already be open.");
        options ??= new DatabaseLexicalSearchOptions();
        options.Validate();
        if (lexicalQuery.Normalization < 0)
            throw new ArgumentOutOfRangeException(nameof(lexicalQuery.Normalization));

        var selected = fields ?? Array.Empty<SearchFieldWeight>();
        var weights = ResolveWeights(lexicalQuery.Fields, selected);
        var weightsSql = string.Join(", ", weights.Select(value => value.ToString("R", CultureInfo.InvariantCulture)));
        var table = PgVectorSemanticSearch.QuoteIdentifierPath(lexicalQuery.Table);
        var itemKey = $"t.{PgVectorSemanticSearch.QuoteIdentifier(lexicalQuery.ItemKeyColumn)}";
        var searchVector = $"t.{PgVectorSemanticSearch.QuoteIdentifier(lexicalQuery.SearchVectorColumn)}";
        var queryFunction = lexicalQuery.QueryMode switch
        {
            PgTextSearchQueryMode.WebSearch => "websearch_to_tsquery",
            PgTextSearchQueryMode.Plain => "plainto_tsquery",
            PgTextSearchQueryMode.Phrase => "phraseto_tsquery",
            PgTextSearchQueryMode.Native => "to_tsquery",
            _ => throw new ArgumentOutOfRangeException(nameof(lexicalQuery.QueryMode))
        };
        var rankFunction = lexicalQuery.RankMode == PgTextSearchRankMode.CoverDensity ? "ts_rank_cd" : "ts_rank";
        var rankProfile = lexicalQuery.RankMode == PgTextSearchRankMode.CoverDensity
            ? LexicalScoringProfiles.PostgreSqlTsRankCd
            : LexicalScoringProfiles.PostgreSqlTsRank;

        var portableFilter = SearchFilterSqlCompiler.Compile(
            lexicalQuery.Filter,
            logical => $"t.{PgVectorSemanticSearch.QuoteIdentifier(ResolveFilterColumn(lexicalQuery.FilterColumns, logical))}");
        var where = new List<string> { $"{searchVector} @@ q.value" };
        if (!string.IsNullOrWhiteSpace(portableFilter.Sql)) where.Add(portableFilter.Sql);
        if (!string.IsNullOrWhiteSpace(lexicalQuery.AdditionalWhereSql)) where.Add($"({lexicalQuery.AdditionalWhereSql})");
        var whereSql = string.Join(" AND ", where);
        var rank = $"{rankFunction}(ARRAY[{weightsSql}]::real[], {searchVector}, q.value, {lexicalQuery.Normalization})";

        var sql = $"""
            WITH q AS (
                SELECT {queryFunction}(CAST(@ote_lexical_config AS regconfig), @ote_lexical_query) AS value
            ), ranked AS MATERIALIZED (
                SELECT {itemKey} AS ote_item_key,
                       {rank} AS ote_rank
                FROM {table} AS t, q
                WHERE {whereSql}
            )
            SELECT ote_item_key, ote_rank
            FROM ranked
            WHERE ote_rank > 0
            ORDER BY ote_rank DESC
            LIMIT @ote_lexical_top
            """;

        await using var command = new NpgsqlCommand(sql, connection, transaction);
        command.Parameters.AddWithValue("ote_lexical_config", lexicalQuery.TextSearchConfiguration);
        command.Parameters.AddWithValue("ote_lexical_query", query);
        command.Parameters.AddWithValue("ote_lexical_top", options.Top);
        foreach (var parameter in portableFilter.Parameters)
            command.Parameters.AddWithValue(parameter.Name, parameter.Value ?? DBNull.Value);
        configureFilterParameters?.Invoke(command);

        var results = new List<LexicalSearchResult<TKey>>(options.Top);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            results.Add(new LexicalSearchResult<TKey>
            {
                Item = PgVectorSemanticSearch.ReadKey<TKey>(reader.GetValue(0)),
                Score = Convert.ToSingle(reader.GetValue(1), CultureInfo.InvariantCulture),
                Fields = Array.Empty<LexicalFieldMatch>(),
                Scoring = new LexicalScoringInfo(rankProfile, 1)
                {
                    Provider = "PostgreSQL",
                    Mode = $"{lexicalQuery.QueryMode}/{lexicalQuery.RankMode}"
                }
            });
        }

        return new DatabaseLexicalSearchResult<TKey>
        {
            Results = results,
            Retrieval = new LexicalCandidateRetrievalInfo
            {
                Provider = "PostgreSQL full-text search",
                Mode = $"{lexicalQuery.QueryMode}/{lexicalQuery.RankMode}",
                RequestedCount = options.Top,
                ReturnedCount = results.Count
            }
        };
    }

    private static float[] ResolveWeights(IReadOnlyList<PgTextSearchField> mappings, IReadOnlyList<SearchFieldWeight> selected)
    {
        var weights = new float[4];
        if (selected.Count == 0)
        {
            if (mappings.Count == 0)
                Array.Fill(weights, 1f);
            else
                foreach (var mapping in mappings)
                    weights[(int)mapping.Weight] = Math.Max(weights[(int)mapping.Weight], 1f);
            return weights;
        }

        var map = mappings.ToDictionary(field => field.Name, field => field.Weight, StringComparer.Ordinal);
        foreach (var field in selected)
        {
            if (field.Weight < 0 || !float.IsFinite(field.Weight))
                throw new ArgumentOutOfRangeException(nameof(selected), $"Lexical field '{field.Name}' has an invalid weight.");
            if (!map.TryGetValue(field.Name, out var label))
                throw new ArgumentException($"No PostgreSQL text-search weight label was configured for logical field '{field.Name}'.", nameof(selected));
            weights[(int)label] = Math.Max(weights[(int)label], field.Weight);
        }
        return weights;
    }

    private static string ResolveFilterColumn(IReadOnlyDictionary<string, string> columns, string logical)
    {
        if (!columns.TryGetValue(logical, out var physical) || string.IsNullOrWhiteSpace(physical))
            throw new ArgumentException($"No PostgreSQL lexical filter-column mapping was configured for logical field '{logical}'.");
        return physical;
    }
}
