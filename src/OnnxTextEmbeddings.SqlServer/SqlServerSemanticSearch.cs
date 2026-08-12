using System.Globalization;
using Microsoft.Data;
using Microsoft.Data.SqlClient;
using Microsoft.Data.SqlTypes;
using Microsoft.Extensions.DependencyInjection;

namespace OnnxTextEmbeddings.SqlServer;

public static class SqlServerVectorLimits
{
    public const int MaximumDimensions = 1998;
}

public enum SqlServerVectorSearchMode
{
    Exact = 1,
    Approximate = 2
}

public sealed record SqlServerVectorCapabilities
{
    public required bool SupportsVectorType { get; init; }
    public required bool SupportsApproximateSearch { get; init; }
    public required bool PreviewFeaturesEnabled { get; init; }
    public int MaximumDimensions => SqlServerVectorLimits.MaximumDimensions;
}

public sealed record SqlServerCandidateQuery
{
    public required string Table { get; init; }
    public required string ItemKeyColumn { get; init; }
    public required string FieldNameColumn { get; init; }
    public required string FingerprintColumn { get; init; }
    public required string VectorColumn { get; init; }
    public required string RecordJsonColumn { get; init; }
    public string? FieldWeightColumn { get; init; }
    public string? AdditionalWhereSql { get; init; }
    public int? VectorDimensions { get; init; }
    public SqlServerVectorSearchMode SearchMode { get; init; } = SqlServerVectorSearchMode.Exact;
}

public static class SqlServerEmbeddingExtensions
{
    public static TextEmbedding ToSqlServerVectorSpace(
        this TextEmbedding embedding,
        int? dimensions = null,
        EmbeddingDimensionReductionStrategy strategy = EmbeddingDimensionReductionStrategy.Auto)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        var target = ResolveTarget(embedding.Vector.Dimensions, dimensions);
        return embedding.ReduceDimensions(target, EmbeddingVectorFormat.Float32, strategy);
    }

    public static QueryEmbedding ToSqlServerVectorSpace(
        this QueryEmbedding query,
        int? dimensions = null,
        EmbeddingDimensionReductionStrategy strategy = EmbeddingDimensionReductionStrategy.Auto)
    {
        ArgumentNullException.ThrowIfNull(query);
        var target = ResolveTarget(query.Vector.Dimensions, dimensions);
        return query.ReduceDimensions(target, EmbeddingVectorFormat.Float32, strategy);
    }

    private static int ResolveTarget(int sourceDimensions, int? requested)
    {
        var target = requested ?? Math.Min(sourceDimensions, SqlServerVectorLimits.MaximumDimensions);
        if (target <= 0 || target > SqlServerVectorLimits.MaximumDimensions)
            throw new ArgumentOutOfRangeException(nameof(requested), $"SQL Server VECTOR supports 1..{SqlServerVectorLimits.MaximumDimensions} dimensions.");
        if (target > sourceDimensions)
            throw new ArgumentOutOfRangeException(nameof(requested), "SQL Server vector preparation cannot expand embedding dimensionality.");
        return target;
    }
}

/// <summary>
/// Uses SQL Server 2025/Azure SQL native vector search for candidate retrieval. Exact kNN is the default; preview
/// approximate search is opt-in. Final semantic ranking remains the canonical core DefaultV1 implementation.
/// </summary>
public sealed class SqlServerSemanticSearch(ISemanticCandidateReranker reranker)
{
    public async Task<DatabaseSemanticSearchResult<TKey>> SearchAsync<TKey>(
        SqlConnection connection,
        QueryEmbedding query,
        SqlServerCandidateQuery candidateQuery,
        DatabaseSemanticSearchOptions? options = null,
        Action<SqlCommand>? configureFilterParameters = null,
        SqlTransaction? transaction = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        options ??= new DatabaseSemanticSearchOptions();
        var databaseQuery = query.ToSqlServerVectorSpace(candidateQuery.VectorDimensions);
        var candidates = await FindCandidatesAsync<TKey>(
            connection,
            databaseQuery,
            candidateQuery,
            options,
            configureFilterParameters,
            transaction,
            cancellationToken).ConfigureAwait(false);
        return await reranker.RerankAsync(databaseQuery, candidates, options, cancellationToken).ConfigureAwait(false);
    }

    public async Task<SemanticCandidateBatch<TKey>> FindCandidatesAsync<TKey>(
        SqlConnection connection,
        QueryEmbedding databaseQuery,
        SqlServerCandidateQuery candidateQuery,
        DatabaseSemanticSearchOptions? options = null,
        Action<SqlCommand>? configureFilterParameters = null,
        SqlTransaction? transaction = null,
        CancellationToken cancellationToken = default)
        where TKey : notnull
    {
        ArgumentNullException.ThrowIfNull(connection);
        ArgumentNullException.ThrowIfNull(databaseQuery);
        ArgumentNullException.ThrowIfNull(candidateQuery);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The SqlConnection must already be open.");
        if (databaseQuery.Vector.Dimensions > SqlServerVectorLimits.MaximumDimensions)
            throw new ArgumentOutOfRangeException(nameof(databaseQuery), "The query must be reduced to a SQL Server-compatible vector space before candidate retrieval.");
        options ??= new DatabaseSemanticSearchOptions();
        var candidateCount = options.ResolveCandidateCount();

        var weight = candidateQuery.FieldWeightColumn is null
            ? "CAST(1.0 AS real)"
            : $"t.{QuoteIdentifier(candidateQuery.FieldWeightColumn)}";
        var extraWhere = string.IsNullOrWhiteSpace(candidateQuery.AdditionalWhereSql)
            ? string.Empty
            : $" AND ({candidateQuery.AdditionalWhereSql})";
        var table = QuoteIdentifierPath(candidateQuery.Table);
        var vector = QuoteIdentifier(candidateQuery.VectorColumn);
        var itemKey = QuoteIdentifier(candidateQuery.ItemKeyColumn);
        var fieldName = QuoteIdentifier(candidateQuery.FieldNameColumn);
        var fingerprint = QuoteIdentifier(candidateQuery.FingerprintColumn);
        var recordJson = QuoteIdentifier(candidateQuery.RecordJsonColumn);

        string sql;
        if (candidateQuery.SearchMode == SqlServerVectorSearchMode.Approximate)
        {
            sql = $"""
                SELECT TOP (@ote_candidate_count) WITH APPROXIMATE
                       t.{itemKey},
                       t.{fieldName},
                       t.{recordJson},
                       {weight},
                       1.0 - r.distance AS native_similarity
                FROM VECTOR_SEARCH(
                    TABLE = {table} AS t,
                    COLUMN = {vector},
                    SIMILAR_TO = @ote_query,
                    METRIC = 'cosine'
                ) AS r
                WHERE t.{fingerprint} = @ote_fingerprint{extraWhere}
                ORDER BY r.distance
                """;
        }
        else
        {
            var distance = $"VECTOR_DISTANCE('cosine', t.{vector}, @ote_query)";
            sql = $"""
                SELECT TOP (@ote_candidate_count)
                       t.{itemKey},
                       t.{fieldName},
                       t.{recordJson},
                       {weight},
                       1.0 - {distance} AS native_similarity
                FROM {table} AS t
                WHERE t.{fingerprint} = @ote_fingerprint{extraWhere}
                ORDER BY {distance}
                """;
        }

        await using var command = new SqlCommand(sql, connection, transaction);
        command.Parameters.Add(new SqlParameter("@ote_query", SqlDbTypeExtensions.Vector)
        {
            Value = new SqlVector<float>(databaseQuery.Vector.ToFloat32())
        });
        command.Parameters.AddWithValue("@ote_fingerprint", databaseQuery.Identity.EmbeddingSpaceFingerprint);
        command.Parameters.AddWithValue("@ote_candidate_count", candidateCount);
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
                Provider = "SQL Server/Azure SQL",
                Mode = candidateQuery.SearchMode.ToString(),
                RequestedCandidateCount = candidateCount,
                ReturnedCandidateCount = results.Count,
                Approximate = candidateQuery.SearchMode == SqlServerVectorSearchMode.Approximate
            }
        };
    }

    public static async Task<SqlServerVectorCapabilities> GetCapabilitiesAsync(
        SqlConnection connection,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(connection);
        if (connection.State != System.Data.ConnectionState.Open)
            throw new InvalidOperationException("The SqlConnection must already be open.");

        var vectorSupported = false;
        try
        {
            await using var probe = new SqlCommand(
                "SELECT VECTOR_DISTANCE('cosine', CAST('[1,0]' AS VECTOR(2)), CAST('[1,0]' AS VECTOR(2)))",
                connection);
            _ = await probe.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            vectorSupported = true;
        }
        catch (SqlException)
        {
        }

        var previewEnabled = false;
        try
        {
            await using var preview = new SqlCommand(
                "SELECT CAST(value AS int) FROM sys.database_scoped_configurations WHERE name = 'PREVIEW_FEATURES'",
                connection);
            var value = await preview.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
            previewEnabled = value is not null and not DBNull && Convert.ToInt32(value, CultureInfo.InvariantCulture) != 0;
        }
        catch (SqlException)
        {
        }

        var approximate = false;
        try
        {
            await using var indexes = new SqlCommand(
                "SELECT CASE WHEN OBJECT_ID('sys.vector_indexes', 'V') IS NULL THEN 0 ELSE 1 END",
                connection);
            approximate = Convert.ToInt32(await indexes.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false), CultureInfo.InvariantCulture) != 0;
        }
        catch (SqlException)
        {
        }

        return new SqlServerVectorCapabilities
        {
            SupportsVectorType = vectorSupported,
            SupportsApproximateSearch = approximate,
            PreviewFeaturesEnabled = previewEnabled
        };
    }

    private static string QuoteIdentifier(string identifier)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identifier);
        return $"[{identifier.Replace("]", "]]", StringComparison.Ordinal)}]";
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

public static class SqlServerServiceCollectionExtensions
{
    public static IServiceCollection AddOnnxTextEmbeddingsSqlServer(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.AddSingleton<SqlServerSemanticSearch>();
        return services;
    }
}
