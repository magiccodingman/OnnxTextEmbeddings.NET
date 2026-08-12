using System.Globalization;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;

namespace OnnxTextEmbeddings.SqliteVec;

public enum SqliteVecStorageKind
{
    Float32 = 1,
    Int8 = 2
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

/// <summary>Native sqlite-vec KNN candidate retrieval followed by the shared core semantic reranker.</summary>
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
        var where = new List<string>
        {
            $"{vectorColumn} MATCH {vectorConstructor}($ote_query)",
            "k = $ote_candidate_count",
            $"{fingerprint} = $ote_fingerprint"
        };
        if (!string.IsNullOrWhiteSpace(portableFilter.Sql)) where.Add(portableFilter.Sql);
        if (!string.IsNullOrWhiteSpace(candidateQuery.AdditionalWhereSql)) where.Add($"({candidateQuery.AdditionalWhereSql})");
        if (candidateQuery.IncludeFields is { } fields)
        {
            if (fields.Count == 0)
                where.Add("1 = 0");
            else
                where.Add($"{fieldName} IN ({string.Join(", ", fields.Select((_, index) => $"$ote_field_{index}"))})");
        }

        var sql = $"""
            SELECT {itemKey},
                   {fieldName},
                   {recordJson},
                   {weight},
                   1.0 - distance AS native_similarity
            FROM {table}
            WHERE {string.Join(" AND ", where)}
            ORDER BY distance
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.AddWithValue("$ote_fingerprint", query.Identity.EmbeddingSpaceFingerprint);
        command.Parameters.AddWithValue("$ote_candidate_count", candidateCount);
        var vector = candidateQuery.StorageKind == SqliteVecStorageKind.Int8
            ? query.Vector.ConvertTo(EmbeddingVectorFormat.Int8)
            : query.Vector.ConvertTo(EmbeddingVectorFormat.Float32);
        command.Parameters.Add("$ote_query", SqliteType.Blob).Value = vector.Data;
        foreach (var parameter in portableFilter.Parameters)
            command.Parameters.AddWithValue("$" + parameter.Name, parameter.Value ?? DBNull.Value);
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
                Mode = candidateQuery.StorageKind.ToString(),
                RequestedCandidateCount = candidateCount,
                ReturnedCandidateCount = results.Count,
                Approximate = false
            }
        };
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
