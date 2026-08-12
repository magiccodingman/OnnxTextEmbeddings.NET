using System.Globalization;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;

namespace OnnxTextEmbeddings.SqliteVec;

public enum SqliteVecStorageKind
{
    Float32 = 1,
    Int8 = 2
}

public sealed record SqliteVecCapabilities
{
    public required string Version { get; init; }
    public bool SupportsFloat32 => true;
    public bool SupportsInt8 => true;
}

public sealed record SqliteVecCandidateQuery
{
    public required string Table { get; init; }
    public required string ItemKeyColumn { get; init; }
    public required string FieldNameColumn { get; init; }
    public required string FingerprintColumn { get; init; }
    public required string VectorColumn { get; init; }
    public required string RecordJsonColumn { get; init; }
    public string? FieldWeightColumn { get; init; }
    public string? AdditionalWhereSql { get; init; }
    public SearchFilter? Filter { get; init; }
    public IReadOnlyDictionary<string, string> FilterColumns { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal);
    public IReadOnlyList<string>? IncludeFields { get; init; }
    public IReadOnlyDictionary<string, float>? QueryFieldWeights { get; init; }
    public SqliteVecStorageKind StorageKind { get; init; } = SqliteVecStorageKind.Float32;
}

public static class SqliteVecConnectionExtensions
{
    /// <summary>Loads the sqlite-vec native extension supplied by the exact-version-pinned sqlite-vec dependency.</summary>
    public static void LoadOnnxTextEmbeddingsSqliteVec(this SqliteConnection connection)
    {
        ArgumentNullException.ThrowIfNull(connection);
        if (connection.State != System.Data.ConnectionState.Closed)
            throw new InvalidOperationException("sqlite-vec must be loaded before opening the SQLite connection.");
        connection.LoadVector();
    }

    public static async Task<SqliteVecCapabilities> GetSqliteVecCapabilitiesAsync(
        this SqliteConnection connection,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(connection);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The SQLite connection must already be open.");
        await using var command = connection.CreateCommand();
        command.CommandText = "SELECT vec_version()";
        try
        {
            var value = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            return new SqliteVecCapabilities { Version = Convert.ToString(value, CultureInfo.InvariantCulture) ?? "unknown" };
        }
        catch (SqliteException ex)
        {
            throw new InvalidOperationException(
                "sqlite-vec is not loaded on this connection. Call LoadOnnxTextEmbeddingsSqliteVec() before Open/OpenAsync().",
                ex);
        }
    }
}

/// <summary>
/// sqlite-vec semantic candidate retrieval followed by the shared core reranker. Filter shapes that vec0 can safely
/// push into its KNN metadata planner use MATCH/KNN directly; richer portable filters fall back to an exact filtered
/// scalar cosine scan so filter semantics are preserved instead of being weakened to fit vec0's restricted grammar.
/// </summary>
public sealed class SqliteVecSemanticSearch(ISemanticCandidateReranker reranker)
{
    public async Task<DatabaseSemanticSearchResult<TKey>> SearchAsync<TKey>(
        SqliteConnection connection,
        QueryEmbedding query,
        SqliteVecCandidateQuery candidateQuery,
        DatabaseSemanticSearchOptions? options = null,
        Action<SqliteCommand>? configureFilterParameters = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        options ??= new DatabaseSemanticSearchOptions();
        var candidates = await FindCandidatesAsync<TKey>(
            connection,
            query,
            candidateQuery,
            options,
            configureFilterParameters,
            cancellationToken).ConfigureAwait(false);
        return await reranker.RerankAsync(query, candidates, options, cancellationToken).ConfigureAwait(false);
    }

    public async Task<SemanticCandidateBatch<TKey>> FindCandidatesAsync<TKey>(
        SqliteConnection connection,
        QueryEmbedding query,
        SqliteVecCandidateQuery candidateQuery,
        DatabaseSemanticSearchOptions? options = null,
        Action<SqliteCommand>? configureFilterParameters = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(candidateQuery);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The SqliteConnection must already be open.");
        _ = await connection.GetSqliteVecCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
        options ??= new DatabaseSemanticSearchOptions();
        var candidateCount = options.ResolveCandidateCount();

        var table = QuoteIdentifier(candidateQuery.Table);
        var itemKey = QuoteIdentifier(candidateQuery.ItemKeyColumn);
        var fieldName = QuoteIdentifier(candidateQuery.FieldNameColumn);
        var fingerprint = QuoteIdentifier(candidateQuery.FingerprintColumn);
        var vectorColumn = QuoteIdentifier(candidateQuery.VectorColumn);
        var recordJson = QuoteIdentifier(candidateQuery.RecordJsonColumn);
        var weight = candidateQuery.FieldWeightColumn is null
            ? "CAST(1.0 AS REAL)"
            : QuoteIdentifier(candidateQuery.FieldWeightColumn);
        var vectorConstructor = candidateQuery.StorageKind == SqliteVecStorageKind.Int8 ? "vec_int8" : "vec_f32";

        var portableFilter = SearchFilterSqlCompiler.Compile(
            candidateQuery.Filter,
            logical => QuoteIdentifier(ResolveFilterColumn(candidateQuery.FilterColumns, logical)));

        var reservedKnnColumns = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            candidateQuery.FingerprintColumn
        };
        if (candidateQuery.IncludeFields is { Count: 1 })
            reservedKnnColumns.Add(candidateQuery.FieldNameColumn);

        var knnFilterSupported = TryCompileKnnFilter(
            candidateQuery.Filter,
            candidateQuery.FilterColumns,
            reservedKnnColumns,
            out var knnFilter);
        var useKnn = knnFilterSupported
            && string.IsNullOrWhiteSpace(candidateQuery.AdditionalWhereSql)
            && candidateQuery.IncludeFields is null or { Count: <= 1 };

        string sql;
        string retrievalMode;
        CompiledSearchFilter filterToBind;
        if (useKnn)
        {
            var where = new List<string>
            {
                $"{vectorColumn} MATCH {vectorConstructor}($ote_query)",
                "k = $ote_candidate_count",
                $"{fingerprint} = $ote_fingerprint"
            };
            if (!string.IsNullOrWhiteSpace(knnFilter.Sql))
                where.Add(knnFilter.Sql);
            if (candidateQuery.IncludeFields is { Count: 1 } oneField)
                where.Add($"{fieldName} = $ote_field_0");

            sql = $"""
                SELECT {itemKey},
                       {fieldName},
                       {recordJson},
                       {weight},
                       1.0 - distance AS native_similarity
                FROM {table}
                WHERE {string.Join(" AND ", where)}
                ORDER BY distance
                """;
            retrievalMode = $"{candidateQuery.StorageKind}/KNN";
            filterToBind = knnFilter;
        }
        else
        {
            var where = new List<string> { $"{fingerprint} = $ote_fingerprint" };
            if (!string.IsNullOrWhiteSpace(portableFilter.Sql))
                where.Add(portableFilter.Sql);
            if (!string.IsNullOrWhiteSpace(candidateQuery.AdditionalWhereSql))
                where.Add($"({candidateQuery.AdditionalWhereSql})");
            if (candidateQuery.IncludeFields is { } fields)
            {
                if (fields.Count == 0)
                    where.Add("1 = 0");
                else
                    where.Add($"{fieldName} IN ({string.Join(", ", fields.Select((_, index) => $"$ote_field_{index}"))})");
            }

            var distance = $"vec_distance_cosine({vectorColumn}, {vectorConstructor}($ote_query))";
            sql = $"""
                WITH filtered AS MATERIALIZED (
                    SELECT {itemKey} AS ote_item_key,
                           {fieldName} AS ote_field_name,
                           {recordJson} AS ote_record_json,
                           {weight} AS ote_field_weight,
                           {distance} AS ote_distance
                    FROM {table}
                    WHERE {string.Join(" AND ", where)}
                )
                SELECT ote_item_key,
                       ote_field_name,
                       ote_record_json,
                       ote_field_weight,
                       1.0 - ote_distance AS native_similarity
                FROM filtered
                ORDER BY ote_distance
                LIMIT $ote_candidate_count
                """;
            retrievalMode = $"{candidateQuery.StorageKind}/FilteredExactScan";
            filterToBind = portableFilter;
        }

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.AddWithValue("$ote_fingerprint", query.Identity.EmbeddingSpaceFingerprint);
        command.Parameters.AddWithValue("$ote_candidate_count", candidateCount);
        var vector = candidateQuery.StorageKind == SqliteVecStorageKind.Int8
            ? query.Vector.ConvertTo(EmbeddingVectorFormat.Int8)
            : query.Vector.ConvertTo(EmbeddingVectorFormat.Float32);
        command.Parameters.Add("$ote_query", SqliteType.Blob).Value = vector.Data;
        foreach (var parameter in filterToBind.Parameters)
            command.Parameters.AddWithValue("@" + parameter.Name, parameter.Value ?? DBNull.Value);
        if (candidateQuery.IncludeFields is { } included)
        {
            for (var index = 0; index < included.Count; index++)
                command.Parameters.AddWithValue($"$ote_field_{index}", included[index]);
        }
        configureFilterParameters?.Invoke(command);

        var results = new List<SemanticCandidate<TKey>>(candidateCount);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            var name = reader.GetString(1);
            var fieldWeight = Convert.ToSingle(reader.GetValue(3), CultureInfo.InvariantCulture);
            if (candidateQuery.QueryFieldWeights is { } queryWeights && queryWeights.TryGetValue(name, out var queryWeight))
                fieldWeight *= queryWeight;
            results.Add(new SemanticCandidate<TKey>
            {
                ItemKey = ReadKey<TKey>(reader.GetValue(0)),
                FieldName = name,
                Embedding = EmbeddingSerializer.DeserializeJson(reader.GetString(2)),
                FieldWeight = fieldWeight,
                NativeSimilarity = Convert.ToSingle(reader.GetValue(4), CultureInfo.InvariantCulture)
            });
        }

        return new SemanticCandidateBatch<TKey>
        {
            Candidates = results,
            Retrieval = new SemanticCandidateRetrievalInfo
            {
                Provider = "SQLite/sqlite-vec",
                Mode = retrievalMode,
                RequestedCandidateCount = candidateCount,
                ReturnedCandidateCount = results.Count,
                Approximate = false
            }
        };
    }

    private static bool TryCompileKnnFilter(
        SearchFilter? filter,
        IReadOnlyDictionary<string, string> columns,
        HashSet<string> usedPhysicalColumns,
        out CompiledSearchFilter compiled)
    {
        var parameters = new List<SearchFilterSqlParameter>();
        var clauses = new List<string>();
        var supported = filter is null || TryAppend(filter);
        compiled = supported
            ? new CompiledSearchFilter(string.Join(" AND ", clauses), parameters)
            : new CompiledSearchFilter(string.Empty, Array.Empty<SearchFilterSqlParameter>());
        return supported;

        bool TryAppend(SearchFilter current)
        {
            if (current is SearchLogicalFilter { Operator: SearchLogicalOperator.And } logical)
                return logical.Filters.All(TryAppend);
            if (current is not SearchComparisonFilter comparison || comparison.Value is null)
                return false;
            if (comparison.Operator == SearchComparisonOperator.NotEqual)
                return false; // NULL != value is true in the portable evaluator but not in SQL three-valued logic.

            var physical = ResolveFilterColumn(columns, comparison.Field);
            if (!usedPhysicalColumns.Add(physical))
                return false; // vec0 permits at most one KNN metadata constraint per metadata column.

            var op = comparison.Operator switch
            {
                SearchComparisonOperator.Equal => "=",
                SearchComparisonOperator.GreaterThan => ">",
                SearchComparisonOperator.GreaterThanOrEqual => ">=",
                SearchComparisonOperator.LessThan => "<",
                SearchComparisonOperator.LessThanOrEqual => "<=",
                _ => null
            };
            if (op is null)
                return false;

            var name = $"ote_filter_{parameters.Count}";
            parameters.Add(new SearchFilterSqlParameter(name, comparison.Value));
            clauses.Add($"{QuoteIdentifier(physical)} {op} @{name}");
            return true;
        }
    }

    private static string ResolveFilterColumn(IReadOnlyDictionary<string, string> columns, string logical)
    {
        if (!columns.TryGetValue(logical, out var physical) || string.IsNullOrWhiteSpace(physical))
            throw new ArgumentException($"No SQLite filter-column mapping was configured for logical field '{logical}'.");
        return physical;
    }

    internal static string QuoteIdentifier(string identifier)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identifier);
        return $"\"{identifier.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";
    }

    internal static TKey ReadKey<TKey>(object value) where TKey : notnull
    {
        if (value is TKey typed)
            return typed;
        if (typeof(TKey) == typeof(Guid) && value is string text && Guid.TryParse(text, out var guid))
            return (TKey)(object)guid;
        return (TKey)Convert.ChangeType(value, typeof(TKey), CultureInfo.InvariantCulture);
    }
}

public static class SqliteVecServiceCollectionExtensions
{
    public static IServiceCollection AddOnnxTextEmbeddingsSqliteVec(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.AddSingleton<SqliteVecSemanticSearch>();
        services.AddSingleton<SqliteFts5LexicalSearch>();
        services.AddSingleton<SqliteVecAdvancedSearch>();
        return services;
    }
}
