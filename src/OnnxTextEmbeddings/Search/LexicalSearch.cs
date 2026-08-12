using System.Text;

namespace OnnxTextEmbeddings;

public static class LexicalScoringProfiles
{
    public const string Bm25V1 = "BM25-v1";
}

public sealed record LexicalField(string Name, string Text, float Weight = 1f)
{
    public static LexicalField Create(string name, string? text, float weight = 1f) => new(name, text ?? string.Empty, weight);
}

public sealed class LexicalSearchRequest
{
    public int Top { get; set; } = 10;
    public float K1 { get; set; } = 1.2f;
    public float B { get; set; } = 0.75f;

    internal void Validate()
    {
        if (Top <= 0) throw new ArgumentOutOfRangeException(nameof(Top));
        if (K1 <= 0) throw new ArgumentOutOfRangeException(nameof(K1));
        if (B is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(B));
    }
}

public sealed record LexicalScoringInfo(string ProfileId, int ProfileVersion, float K1, float B);

public sealed record LexicalFieldMatch
{
    public required string Name { get; init; }
    public required float Weight { get; init; }
    public required float Score { get; init; }
}

public sealed record LexicalSearchResult<T>
{
    public required T Item { get; init; }
    public required float Score { get; init; }
    public required IReadOnlyList<LexicalFieldMatch> Fields { get; init; }
    public required LexicalScoringInfo Scoring { get; init; }
}

public interface ILexicalSearch
{
    Task<IReadOnlyList<LexicalSearchResult<T>>> SearchAsync<T>(
        string query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<LexicalField>> fields,
        LexicalSearchRequest? request = null,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Deterministic in-memory BM25 implementation for small/medium working sets. Database packages use their native
/// lexical engines instead, but expose the same lexical-search intent and field-weight model.
/// </summary>
public sealed class InMemoryLexicalSearch : ILexicalSearch
{
    public Task<IReadOnlyList<LexicalSearchResult<T>>> SearchAsync<T>(
        string query,
        IEnumerable<T> items,
        Func<T, IReadOnlyList<LexicalField>> fields,
        LexicalSearchRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentNullException.ThrowIfNull(items);
        ArgumentNullException.ThrowIfNull(fields);
        request ??= new LexicalSearchRequest();
        request.Validate();

        var queryTerms = Tokenize(query).Distinct(StringComparer.Ordinal).ToArray();
        if (queryTerms.Length == 0)
            return Task.FromResult<IReadOnlyList<LexicalSearchResult<T>>>(Array.Empty<LexicalSearchResult<T>>());

        var documents = new List<PreparedDocument<T>>();
        foreach (var item in items)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var preparedFields = fields(item).Select(PrepareField).ToArray();
            documents.Add(new PreparedDocument<T>(item, preparedFields, preparedFields.Sum(field => field.TokenCount)));
        }
        if (documents.Count == 0)
            return Task.FromResult<IReadOnlyList<LexicalSearchResult<T>>>(Array.Empty<LexicalSearchResult<T>>());

        var averageLength = Math.Max(1d, documents.Average(document => (double)document.TokenCount));
        var documentFrequency = queryTerms.ToDictionary(
            term => term,
            term => documents.Count(document => document.Fields.Any(field => field.Weight > 0 && field.Terms.ContainsKey(term))),
            StringComparer.Ordinal);

        var results = new List<LexicalSearchResult<T>>(documents.Count);
        foreach (var document in documents)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var fieldScores = new double[document.Fields.Length];
            var totalScore = 0d;

            foreach (var term in queryTerms)
            {
                var weightedTf = 0d;
                var fieldTf = new double[document.Fields.Length];
                for (var i = 0; i < document.Fields.Length; i++)
                {
                    var field = document.Fields[i];
                    if (field.Weight <= 0 || !field.Terms.TryGetValue(term, out var count))
                        continue;
                    fieldTf[i] = field.Weight * count;
                    weightedTf += fieldTf[i];
                }
                if (weightedTf <= 0)
                    continue;

                var df = documentFrequency[term];
                var idf = Math.Log(1d + ((documents.Count - df + 0.5d) / (df + 0.5d)));
                var lengthNormalization = request.K1 * (1d - request.B + request.B * document.TokenCount / averageLength);
                var termScore = idf * ((weightedTf * (request.K1 + 1d)) / (weightedTf + lengthNormalization));
                totalScore += termScore;

                for (var i = 0; i < fieldTf.Length; i++)
                {
                    if (fieldTf[i] > 0)
                        fieldScores[i] += termScore * fieldTf[i] / weightedTf;
                }
            }

            if (totalScore <= 0)
                continue;
            results.Add(new LexicalSearchResult<T>
            {
                Item = document.Item,
                Score = (float)totalScore,
                Fields = document.Fields.Select((field, index) => new LexicalFieldMatch
                {
                    Name = field.Name,
                    Weight = field.Weight,
                    Score = (float)fieldScores[index]
                }).Where(match => match.Score > 0).OrderByDescending(match => match.Score).ToArray(),
                Scoring = new LexicalScoringInfo(LexicalScoringProfiles.Bm25V1, 1, request.K1, request.B)
            });
        }

        return Task.FromResult<IReadOnlyList<LexicalSearchResult<T>>>(results
            .OrderByDescending(result => result.Score)
            .Take(request.Top)
            .ToArray());
    }

    private static PreparedField PrepareField(LexicalField field)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(field.Name);
        if (field.Weight < 0)
            throw new ArgumentOutOfRangeException(nameof(field.Weight), "Lexical field weight cannot be negative.");
        var tokens = Tokenize(field.Text);
        return new PreparedField(
            field.Name,
            field.Weight,
            tokens.Length,
            tokens.GroupBy(token => token, StringComparer.Ordinal).ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal));
    }

    private static string[] Tokenize(string text)
    {
        if (string.IsNullOrEmpty(text)) return Array.Empty<string>();
        var terms = new List<string>();
        var builder = new StringBuilder();
        foreach (var rune in text.EnumerateRunes())
        {
            if (Rune.IsLetterOrDigit(rune))
            {
                foreach (var ch in rune.ToString().ToLowerInvariant())
                    builder.Append(ch);
            }
            else if (builder.Length > 0)
            {
                terms.Add(builder.ToString());
                builder.Clear();
            }
        }
        if (builder.Length > 0)
            terms.Add(builder.ToString());
        return terms.ToArray();
    }

    private sealed record PreparedDocument<T>(T Item, PreparedField[] Fields, int TokenCount);
    private sealed record PreparedField(string Name, float Weight, int TokenCount, Dictionary<string, int> Terms);
}
