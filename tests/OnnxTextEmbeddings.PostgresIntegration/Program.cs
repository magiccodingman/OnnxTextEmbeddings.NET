using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using OnnxTextEmbeddings;
using OnnxTextEmbeddings.PgVector;
using Pgvector;

var connectionString = Environment.GetEnvironmentVariable("POSTGRES_CONNECTION_STRING")
    ?? "Host=localhost;Port=5432;Username=postgres;Password=postgres;Database=embeddings";

var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);
dataSourceBuilder.UseVector();
await using var dataSource = dataSourceBuilder.Build();
await using var connection = await dataSource.OpenConnectionAsync();

await using (var command = new NpgsqlCommand("CREATE EXTENSION IF NOT EXISTS vector", connection))
    await command.ExecuteNonQueryAsync();
connection.ReloadTypes();

await using (var command = new NpgsqlCommand("DROP TABLE IF EXISTS onnx_embedding_integration", connection))
    await command.ExecuteNonQueryAsync();
await using (var command = new NpgsqlCommand(
    "CREATE TABLE onnx_embedding_integration (id integer PRIMARY KEY, embedding vector(4) NOT NULL, portable bytea NOT NULL)", connection))
    await command.ExecuteNonQueryAsync();

var first = EmbeddingVector.FromFloat32(new[] { 1f, 0f, 0f, 0f }, EmbeddingVectorFormat.Int4);
var second = EmbeddingVector.FromFloat32(new[] { 0f, 1f, 0f, 0f }, EmbeddingVectorFormat.Int8);
await InsertPortableAsync(1, first);
await InsertPortableAsync(2, second);

var queryVector = EmbeddingVector.FromFloat32(new[] { 0.99f, 0.01f, 0f, 0f }, EmbeddingVectorFormat.Float32);
await using (var command = new NpgsqlCommand(
    "SELECT id, portable FROM onnx_embedding_integration ORDER BY embedding <=> $1 LIMIT 1", connection))
{
    command.Parameters.AddWithValue(queryVector.ToPgVector());
    await using var reader = await command.ExecuteReaderAsync();
    if (!await reader.ReadAsync() || reader.GetInt32(0) != 1)
        throw new InvalidOperationException("pgvector cosine ordering did not return the expected nearest vector.");
    var restored = EmbeddingSerializer.DeserializeVector((byte[])reader[1]);
    if (restored.Format != EmbeddingVectorFormat.Int4 || restored.Dimensions != 4 || EmbeddingVectorMath.CosineSimilarity(first, restored) < 0.999f)
        throw new InvalidOperationException("Portable BYTEA vector did not round-trip correctly.");
}

const string candidateTable = "onnx_semantic_candidates";
await using (var command = new NpgsqlCommand($"DROP TABLE IF EXISTS {candidateTable}", connection))
    await command.ExecuteNonQueryAsync();
await using (var command = new NpgsqlCommand($"""
    CREATE TABLE {candidateTable} (
        item_id text NOT NULL,
        tenant_id integer NOT NULL,
        field_name text NOT NULL,
        fingerprint text NOT NULL,
        embedding vector(4) NOT NULL,
        record_json text NOT NULL,
        field_weight real NOT NULL
    )
    """, connection))
    await command.ExecuteNonQueryAsync();

const string fingerprint = "postgres-integration-space";
await InsertCandidateAsync("backup", 1, "content", TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint, 0), 1f);
await InsertCandidateAsync("backup", 1, "content", TextEmbeddingFor([0.94f, 0.30f, 0f, 0f], fingerprint, 1), 1f);
await InsertCandidateAsync("network", 1, "content", TextEmbeddingFor([0f, 1f, 0f, 0f], fingerprint, 0), 1f);
await InsertCandidateAsync("other-tenant", 2, "content", TextEmbeddingFor([1f, 0f, 0f, 0f], fingerprint, 0), 1f);

const string lexicalTable = "onnx_lexical_items";
await using (var command = new NpgsqlCommand($"DROP TABLE IF EXISTS {lexicalTable}", connection))
    await command.ExecuteNonQueryAsync();
await using (var command = new NpgsqlCommand($"""
    CREATE TABLE {lexicalTable} (
        item_id text PRIMARY KEY,
        tenant_id integer NOT NULL,
        title text NOT NULL,
        body text NOT NULL,
        search_vector tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A') ||
            setweight(to_tsvector('english'::regconfig, coalesce(body, '')), 'D')
        ) STORED
    );
    CREATE INDEX onnx_lexical_items_search_vector_idx ON {lexicalTable} USING GIN(search_vector);
    """, connection))
    await command.ExecuteNonQueryAsync();
await InsertLexicalAsync("backup", 1, "PostgreSQL Backup", "Restore and disaster recovery procedures.");
await InsertLexicalAsync("network", 1, "Networking", "Firewall and routing documentation.");
await InsertLexicalAsync("body-heavy", 1, "Database Operations", "PostgreSQL backup PostgreSQL backup PostgreSQL backup procedures.");
await InsertLexicalAsync("other-tenant", 2, "PostgreSQL Backup", "Exact title in another tenant.");

var services = new ServiceCollection();
services.AddLogging();
services.AddOnnxTextEmbeddings(options => options.Initialization.WarmupOnStartup = false);
services.AddOnnxTextEmbeddingsPgVector();
await using var provider = services.BuildServiceProvider();
var databaseSearch = provider.GetRequiredService<PgVectorSemanticSearch>();
var lexicalSearch = provider.GetRequiredService<PgVectorLexicalSearch>();
var advancedSearch = provider.GetRequiredService<PgVectorAdvancedSearch>();
var query = new QueryEmbedding
{
    Vector = queryVector,
    Identity = Identity(fingerprint),
    SourceTokenCount = 3,
    InputTokenCount = 3
};

var semanticMapping = new PgVectorCandidateQuery
{
    Table = candidateTable,
    ItemKeyColumn = "item_id",
    FieldNameColumn = "field_name",
    FingerprintColumn = "fingerprint",
    VectorColumn = "embedding",
    RecordJsonColumn = "record_json",
    FieldWeightColumn = "field_weight",
    SearchMode = PgVectorSearchMode.Exact,
    FilterColumns = new Dictionary<string, string> { ["TenantId"] = "tenant_id" },
    Filter = SearchFilter.Equal("TenantId", 1)
};
var searchResult = await databaseSearch.SearchAsync<string>(
    connection,
    query,
    semanticMapping,
    new DatabaseSemanticSearchOptions { Top = 1, CandidateCount = 10 });
if (searchResult.Results.Count != 1 || searchResult.Results[0].Item != "backup")
    throw new InvalidOperationException("Filtered pgvector candidate retrieval + DefaultV1 reranking did not return the expected item.");

var lexicalMapping = new PgVectorLexicalQuery
{
    Table = lexicalTable,
    ItemKeyColumn = "item_id",
    SearchVectorColumn = "search_vector",
    Fields =
    [
        new PgTextSearchField("title", PgTextSearchWeight.A),
        new PgTextSearchField("body", PgTextSearchWeight.D)
    ],
    FilterColumns = new Dictionary<string, string> { ["TenantId"] = "tenant_id" },
    Filter = SearchFilter.Equal("TenantId", 1)
};
var lexicalResult = await lexicalSearch.SearchAsync<string>(
    connection,
    "postgresql backup",
    lexicalMapping,
    [SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)],
    new DatabaseLexicalSearchOptions { Top = 2 });
if (lexicalResult.Results.Count == 0 || lexicalResult.Results[0].Item != "backup")
    throw new InvalidOperationException("PostgreSQL native lexical field weighting did not prefer the title match.");

var hybridQuery = SearchQuery.Create("postgresql backup")
    .Where(SearchFilter.Equal("TenantId", 1))
    .Add(SearchRetrievalStage.Semantic(SearchFieldWeight.Create("content", 1)).Candidates(10))
    .Add(SearchRetrievalStage.Lexical(SearchFieldWeight.Create("title", 8), SearchFieldWeight.Create("body", 1)).Candidates(10))
    .Take(2);
var hybrid = await advancedSearch.SearchAsync<string>(
    connection,
    hybridQuery,
    new PgVectorSearchPlan
    {
        Semantic = semanticMapping with { Filter = null },
        Lexical = lexicalMapping with { Filter = null }
    },
    semanticQuery: query);
if (hybrid.Count == 0 || hybrid[0].Item != "backup" || hybrid[0].Contributions.Count != 2)
    throw new InvalidOperationException("PostgreSQL semantic + lexical RRF did not return the jointly supported backup item.");

Console.WriteLine("PASS PostgreSQL pgvector + native full-text + portable filtering + hybrid RRF integration.");

async Task InsertPortableAsync(int id, EmbeddingVector vector)
{
    await using var command = new NpgsqlCommand(
        "INSERT INTO onnx_embedding_integration (id, embedding, portable) VALUES ($1, $2, $3)", connection);
    command.Parameters.AddWithValue(id);
    command.Parameters.AddWithValue(vector.ToPgVector());
    command.Parameters.AddWithValue(EmbeddingSerializer.SerializeVector(vector));
    await command.ExecuteNonQueryAsync();
}

async Task InsertCandidateAsync(string itemId, int tenantId, string field, TextEmbedding embedding, float weight)
{
    await using var command = new NpgsqlCommand($"""
        INSERT INTO {candidateTable}(item_id, tenant_id, field_name, fingerprint, embedding, record_json, field_weight)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, connection);
    command.Parameters.AddWithValue(itemId);
    command.Parameters.AddWithValue(tenantId);
    command.Parameters.AddWithValue(field);
    command.Parameters.AddWithValue(embedding.Identity.EmbeddingSpaceFingerprint);
    command.Parameters.AddWithValue(embedding.Vector.ToPgVector());
    command.Parameters.AddWithValue(EmbeddingSerializer.SerializeJson(embedding));
    command.Parameters.AddWithValue(weight);
    await command.ExecuteNonQueryAsync();
}

async Task InsertLexicalAsync(string itemId, int tenantId, string title, string body)
{
    await using var command = new NpgsqlCommand($"INSERT INTO {lexicalTable}(item_id, tenant_id, title, body) VALUES ($1, $2, $3, $4)", connection);
    command.Parameters.AddWithValue(itemId);
    command.Parameters.AddWithValue(tenantId);
    command.Parameters.AddWithValue(title);
    command.Parameters.AddWithValue(body);
    await command.ExecuteNonQueryAsync();
}

static TextEmbedding TextEmbeddingFor(float[] values, string fingerprint, int chunk) => new()
{
    Vector = EmbeddingVector.FromFloat32(values),
    Identity = Identity(fingerprint),
    Source = new EmbeddingSource
    {
        DocumentTokenCount = 100,
        CharacterRange = new Utf16TextRange(chunk * 10, 10),
        TokenRange = new TokenRange(chunk * 10, 10),
        TokenCount = 10,
        TokenCapacity = 10
    },
    Chunk = new EmbeddingChunkInfo
    {
        Index = chunk,
        Count = 2,
        BoundaryKind = ChunkBoundaryKind.Paragraph,
        InputTokenCount = 10
    }
};

static EmbeddingIdentity Identity(string fingerprint) => new()
{
    ModelId = "integration",
    SourceRevision = "1",
    EmbeddingSpaceFingerprint = fingerprint,
    IsNormalized = true
};
