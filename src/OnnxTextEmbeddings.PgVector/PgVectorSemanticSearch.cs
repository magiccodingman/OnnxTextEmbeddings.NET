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

        NpgsqlTransaction? ownedTransaction = null;
        var effectiveTransaction = transaction;
        try
        {
            if (candidateQuery.SearchMode == PgVectorSearchMode.Exact && effectiveTransaction is null)
            {
                ownedTransaction = (NpgsqlTransaction)await connection.BeginTransactionAsync(cancellationToken).ConfigureAwait(false);
                effectiveTransaction = ownedTransaction;
            }

            if (candidateQuery.SearchMode == PgVectorSearchMode.Exact)
            {
                await using var exact = new NpgsqlCommand(
                    "SET LOCAL enable_indexscan = off; SET LOCAL enable_bitmapscan = off;",
                    connection,
                    effectiveTransaction);
                await exact.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }

            var vectorColumn = QuoteIdentifier(candidateQuery.VectorColumn);
            var distance = $"{vectorColumn} <=> @ote_query";
            var weight = candidateQuery.FieldWeightColumn is null
                ? "CAST(1.0 AS real)"
                : QuoteIdentifier(candidateQuery.FieldWeightColumn);
            var extraWhere = string.IsNullOrWhiteSpace(candidateQuery.AdditionalWhereSql)
                ? string.Empty
                : $" AND ({candidateQuery.AdditionalWhereSql})";

            var sql = $"""
                SELECT {QuoteIdentifier(candidateQuery.ItemKeyColumn)},
                       {QuoteIdentifier(candidateQuery.FieldNameColumn)},
                       {QuoteIdentifier(candidateQuery.RecordJsonColumn)},
                       {weight},
                       1.0 - ({distance}) AS native_similarity
                FROM {QuoteIdentifierPath(candidateQuery.Table)}
                WHERE {QuoteIdentifier(candidateQuery.FingerprintColumn)} = @ote_fingerprint{extraWhere}
                ORDER BY {distance}
                LIMIT @ote_candidate_count
                """;

            await using var command = new NpgsqlCommand(sql, connection, effectiveTransaction);
            command.Parameters.AddWithValue("ote_fingerprint", query.Identity.EmbeddingSpaceFingerprint);
            command.Parameters.AddWithValue("ote_candidate_count", candidateCount);
            command.Parameters.AddWithValue(
                "ote_query",
                candidateQuery.StorageKind == PgVectorStorageKind.HalfVector
                    ? query.Vector.ToPgHalfVector()
                    : query.Vector.ToPgVector());
            configureFilterParameters?.Invoke(command);

            var results = new List<SemanticCandidate<TKey>>(candidateCount);
            await using (var reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false))
            {
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
            }

            if (ownedTransaction is not null)
                await ownedTransaction.CommitAsync(cancellationToken).ConfigureAwait(false);

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
        catch
        {
            if (ownedTransaction is not null)
            {
                try { await ownedTransaction.RollbackAsync(CancellationToken.None).ConfigureAwait(false); }
                catch { }
            }
            throw;
        }
        finally
        {
            if (ownedTransaction is not null)
                await ownedTransaction.DisposeAsync().ConfigureAwait(false);
        }
    }

    private static string QuoteIdentifier(string identifier)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identifier);
        return $"\"{identifier.Replace("\"", "\"\"", StringComparison.Ordinal)}\"";
    }

    private static string QuoteIdentifierPath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var parts = path.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
            throw new ArgumentException("A table name is required.", nameof(path));
        return string.Join('.', parts.Select(QuoteIdentifier));
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

public static class PgVectorServiceCollectionExtensions
{
    public static IServiceCollection AddOnnxTextEmbeddingsPgVector(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.AddSingleton<PgVectorSemanticSearch>();
        return services;
    }
}
