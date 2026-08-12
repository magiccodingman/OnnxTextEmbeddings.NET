using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;

namespace OnnxTextEmbeddings.PgVector;

public enum PgVectorStorageKind
{
    Vector = 1,
    HalfVector = 2
}

public enum PgVectorSearchMode
{
    Exact = 1,
    Approximate = 2
}

/// <summary>Schema mapping and retrieval policy for pgvector-backed chunk candidates.</summary>
public sealed record PgVectorCandidateQuery
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
    public PgVectorStorageKind StorageKind { get; init; } = PgVectorStorageKind.Vector;
    public PgVectorSearchMode SearchMode { get; init; } = PgVectorSearchMode.Exact;
}

/// <summary>
/// Uses PostgreSQL/pgvector for relational filtering and native cosine candidate selection, then delegates final
/// document/field scoring to the canonical core DefaultV1 reranker.
/// </summary>
public sealed class PgVectorSemanticSearch(ISemanticCandidateReranker reranker)
{
    public async Task<DatabaseSemanticSearchResult<TKey>> SearchAsync<TKey>(
        NpgsqlConnection connection,
        QueryEmbedding query,
        PgVectorCandidateQuery candidateQuery,
        DatabaseSemanticSearchOptions? options = null,
        Action<NpgsqlCommand>? configureFilterParameters = null,
        NpgsqlTransaction? transaction = null,
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
            transaction,
            cancellationToken).ConfigureAwait(false);
        return await reranker.RerankAsync(query, candidates, options, cancellationToken).ConfigureAwait(false);
    }

    public async Task<SemanticCandidateBatch<TKey>> FindCandidatesAsync<TKey>(
        NpgsqlConnection connection,
        QueryEmbedding query,
        PgVectorCandidateQuery candidateQuery,
        DatabaseSemanticSearchOptions? options = null,
        Action<NpgsqlCommand>? configureFilterParameters = null,
        NpgsqlTransaction? transaction = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(candidateQuery);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The NpgsqlConnection must already be open.");
        options ??= new DatabaseSemanticSearchOptions();
        var candidateCount = options.ResolveCandidateCount();

        var itemKey = $"t.{QuoteIdentifier(candidateQuery.ItemKeyColumn)}";
        var fieldName = $"t.{QuoteIdentifier(candidateQuery.FieldNameColumn)}";
        var recordJson = $"t.{QuoteIdentifier(candidateQuery.RecordJsonColumn)}";
        var fingerprint = $"t.{QuoteIdentifier(candidateQuery.FingerprintColumn)}";
        var vectorColumn = $"t.{QuoteIdentifier(candidateQuery.VectorColumn)}";
        var table = QuoteIdentifierPath(candidateQuery.Table);
        var distance = $"{vectorColumn} <=> @ote_query";
        var weight = candidateQuery.FieldWeightColumn is null
            ? "CAST(1.0 AS real)"
            : $"t.{QuoteIdentifier(candidateQuery.FieldWeightColumn)}";

        var portableFilter = SearchFilterSqlCompiler.Compile(
            candidateQuery.Filter,
            logical => $"t.{QuoteIdentifier(ResolveFilterColumn(candidateQuery.FilterColumns, logical))}");
        var where = new List<string> { $"{fingerprint} = @ote_fingerprint" };
        if (!string.IsNullOrWhiteSpace(portableFilter.Sql)) where.Add(portableFilter.Sql);
        if (!string.IsNullOrWhiteSpace(candidateQuery.AdditionalWhereSql)) where.Add($"({candidateQuery.AdditionalWhereSql})");
        if (candidateQuery.IncludeFields is { } fields)
        {
            if (fields.Count == 0)
                where.Add("1 = 0");
            else
                where.Add($"{fieldName} IN ({string.Join(", ", fields.Select((_, index) => $"@ote_field_{index}"))})");
        }
        var whereSql = string.Join(" AND ", where);

        var sql = candidateQuery.SearchMode == PgVectorSearchMode.Exact
            ? $"""
                WITH ote_exact AS MATERIALIZED (
                    SELECT {itemKey} AS ote_item_key,
                           {fieldName} AS ote_field_name,
                           {recordJson} AS ote_record_json,
                           {weight} AS ote_field_weight,
                           ({distance}) AS ote_distance
                    FROM {table} AS t
                    WHERE {whereSql}
                )
                SELECT ote_item_key,
                       ote_field_name,
                       ote_record_json,
                       ote_field_weight,
                       1.0 - ote_distance AS native_similarity
                FROM ote_exact
                ORDER BY ote_distance
                LIMIT @ote_candidate_count
                """
            : $"""
                SELECT {itemKey},
                       {fieldName},
                       {recordJson},
                       {weight},
                       1.0 - ({distance}) AS native_similarity
                FROM {table} AS t
                WHERE {whereSql}
                ORDER BY {distance}
                LIMIT @ote_candidate_count
                """;

        await using var command = new NpgsqlCommand(sql, connection, transaction);
        command.Parameters.AddWithValue("ote_fingerprint", query.Identity.EmbeddingSpaceFingerprint);
        command.Parameters.AddWithValue("ote_candidate_count", candidateCount);
        command.Parameters.AddWithValue(
            "ote_query",
            candidateQuery.StorageKind == PgVectorStorageKind.HalfVector
                ? query.Vector.ToPgHalfVector()
                : query.Vector.ToPgVector());
        foreach (var parameter in portableFilter.Parameters)
            command.Parameters.AddWithValue(parameter.Name, parameter.Value ?? DBNull.Value);
        if (candidateQuery.IncludeFields is { } included)
        {
            for (var index = 0; index < included.Count; index++)
                command.Parameters.AddWithValue($"ote_field_{index}", included[index]);
        }
        configureFilterParameters?.Invoke(command);

        var results = new List<SemanticCandidate<TKey>>(candidateCount);
        await using (var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false))
        {
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
        }

        return new SemanticCandidateBatch<TKey>
        {
            Candidates = results,
            Retrieval = new SemanticCandidateRetrievalInfo
            {
                Provider = "PostgreSQL/pgvector",
                Mode = candidateQuery.SearchMode.ToString(),
                RequestedCandidateCount = candidateCount,
                ReturnedCandidateCount = results.Count,
                Approximate = candidateQuery.SearchMode == PgVectorSearchMode.Approximate
            }
        };
    }

    private static string ResolveFilterColumn(IReadOnlyDictionary<string, string> columns, string logical)
    {
        if (!columns.TryGetValue(logical, out var physical) || string.IsNullOrWhiteSpace(physical))
            throw new ArgumentException($"No PostgreSQL filter-column mapping was configured for logical field '{logical}'.");
        return physical;
    }

    internal static string QuoteIdentifier(string identifier)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identifier);
        return $"\"{identifier.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";
    }

    internal static string QuoteIdentifierPath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var parts = path.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
            throw new ArgumentException("A table name is required.", nameof(path));
        return string.Join('.', parts.Select(QuoteIdentifier));
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

public static class PgVectorServiceCollectionExtensions
{
    public static IServiceCollection AddOnnxTextEmbeddingsPgVector(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.AddSingleton<PgVectorSemanticSearch>();
        services.AddSingleton<PgVectorLexicalSearch>();
        services.AddSingleton<PgVectorAdvancedSearch>();
        return services;
    }
}
