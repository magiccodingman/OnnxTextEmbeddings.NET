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

/// <summary>Schema mapping for a sqlite-vec vec0 table containing direct chunk embeddings.</summary>
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
    public SqliteVecStorageKind StorageKind { get; init; } = SqliteVecStorageKind.Float32;
}

public static class SqliteVecConnectionExtensions
{
    /// <summary>Loads the sqlite-vec native extension supplied by the pinned sqlite-vec NuGet package.</summary>
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
/// Keeps sqlite-vec KNN/cosine work inside SQLite and sends only the bounded direct-chunk candidate set through
/// the canonical core DefaultV1 reranker.
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
            throw new InvalidOperationException("The SQLite connection must already be open.");
        _ = await connection.GetSqliteVecCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
        options ??= new DatabaseSemanticSearchOptions();
        var candidateCount = options.ResolveCandidateCount();

        var vectorConstructor = candidateQuery.StorageKind == SqliteVecStorageKind.Int8 ? "vec_int8" : "vec_f32";
        var queryPayload = candidateQuery.StorageKind == SqliteVecStorageKind.Int8
            ? query.Vector.ConvertTo(EmbeddingVectorFormat.Int8).Data
            : query.Vector.ConvertTo(EmbeddingVectorFormat.Float32).Data;
        var weight = candidateQuery.FieldWeightColumn is null
            ? "CAST(1.0 AS REAL)"
            : QuoteIdentifier(candidateQuery.FieldWeightColumn);
        var extraWhere = string.IsNullOrWhiteSpace(candidateQuery.AdditionalWhereSql)
            ? string.Empty
            : $" AND ({candidateQuery.AdditionalWhereSql})";

        var sql = $"""
            SELECT {QuoteIdentifier(candidateQuery.ItemKeyColumn)},
                   {QuoteIdentifier(candidateQuery.FieldNameColumn)},
                   {QuoteIdentifier(candidateQuery.RecordJsonColumn)},
                   {weight},
                   1.0 - distance AS native_similarity
            FROM {QuoteIdentifier(candidateQuery.Table)}
            WHERE {QuoteIdentifier(candidateQuery.VectorColumn)} MATCH {vectorConstructor}($ote_query)
              AND k = $ote_candidate_count
              AND {QuoteIdentifier(candidateQuery.FingerprintColumn)} = $ote_fingerprint{extraWhere}
            ORDER BY distance
            """;

        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        command.Parameters.Add("$ote_query", SqliteType.Blob).Value = queryPayload;
        command.Parameters.Add("$ote_candidate_count", SqliteType.Integer).Value = candidateCount;
        command.Parameters.Add("$ote_fingerprint", SqliteType.Text).Value = query.Identity.EmbeddingSpaceFingerprint;
        configureFilterParameters?.Invoke(command);

        var results = new List<SemanticCandidate<TKey>>(candidateCount);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            results.Add(new SemanticCandidate<TKey>
            {
                ItemKey = ReadKey<TKey>(reader.GetValue(0)),
                FieldName = reader.GetString(1),
                Embedding = EmbeddingSerializer.DeserializeJson(reader.GetString(2)),
                FieldWeight = Convert.ToSingle(reader.GetValue(3), CultureInfo.InvariantCulture),
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

    private static string QuoteIdentifier(string identifier)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identifier);
        return $"\"{identifier.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";
    }

    private static TKey ReadKey<TKey>(object value) where TKey : notnull
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
        return services;
    }
}
