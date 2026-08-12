namespace OnnxTextEmbeddings.Tests;

public sealed class StructuredTextChunkerTests
{
    [Fact]
    public void ShortDocument_RemainsWhole()
    {
        using var tokenizer = new WhitespaceTokenizer();
        var options = new OnnxTextEmbeddingsOptions();
        var chunker = new StructuredTextChunker(tokenizer, options);

        var chunks = chunker.Chunk("hello small world", 10);

        var chunk = Assert.Single(chunks);
        Assert.Equal(ChunkBoundaryKind.WholeDocument, chunk.BoundaryKind);
        Assert.Equal("hello small world", chunk.SourceText);
        Assert.Equal(new Utf16TextRange(0, 17), chunk.CharacterRange);
    }

    [Fact]
    public void Markdown_PreservesHeadingContextOnContinuation()
    {
        using var tokenizer = new WhitespaceTokenizer();
        var options = new OnnxTextEmbeddingsOptions();
        var chunker = new StructuredTextChunker(tokenizer, options);
        var text = "# Backups\n\none two three four five six seven eight nine ten eleven twelve";

        var chunks = chunker.Chunk(text, 7);

        Assert.True(chunks.Count > 1);
        Assert.Contains(chunks.Skip(1), x => x.Context == "Backups");
        Assert.All(chunks, x => Assert.True(x.ModelInput.TokenCount <= 7));
    }

    [Fact]
    public void ConfiguredOverlap_ReusesPriorTokensWithoutExceedingLimit()
    {
        using var tokenizer = new WhitespaceTokenizer();
        var options = new OnnxTextEmbeddingsOptions();
        options.Chunking.ChunkOverlapTokens = 2;
        var chunker = new StructuredTextChunker(tokenizer, options);
        var text = "# Backups\n\none two three four five six seven eight nine ten eleven twelve thirteen fourteen";

        var chunks = chunker.Chunk(text, 8);

        Assert.True(chunks.Count > 1);
        Assert.True(chunks[1].TokenRange.Start < chunks[0].TokenRange.End);
        Assert.All(chunks, x => Assert.True(x.ModelInput.TokenCount <= 8));
    }

    [Fact]
    public void ExactFinalization_ShrinksAndContinuesWhenPlanningUnderestimatesModelTokens()
    {
        using var tokenizer = new WhitespaceTokenizer(emptySpecialTokens: 1, nonEmptySpecialTokens: 2);
        var options = new OnnxTextEmbeddingsOptions();
        var chunker = new StructuredTextChunker(tokenizer, options);
        var text = "# Backups\n\none two three four five six seven eight nine ten eleven twelve";

        var chunks = chunker.Chunk(text, 7);

        Assert.True(chunks.Count > 1);
        Assert.All(chunks, chunk => Assert.True(chunk.ModelInput.TokenCount <= 7));
        Assert.Equal(text, string.Concat(chunks.Select(chunk => chunk.SourceText)));
    }

    [Fact]
    public void CodeFenceHeading_IsNotTreatedAsMarkdownHeading()
    {
        using var tokenizer = new WhitespaceTokenizer();
        var options = new OnnxTextEmbeddingsOptions();
        var chunker = new StructuredTextChunker(tokenizer, options);
        var text = "```md\n# fake\n```\n\nreal words here";

        var chunks = chunker.Chunk(text, 50);

        Assert.Single(chunks);
        Assert.Empty(chunks[0].HeadingPath);
    }

    private sealed class WhitespaceTokenizer(
        int emptySpecialTokens = 1,
        int nonEmptySpecialTokens = 1) : IEmbeddingTokenizer
    {
        public TokenizedSource TokenizeSource(string text)
        {
            var ids = new List<uint>();
            var offsets = new List<TokenOffset>();
            var i = 0;
            while (i < text.Length)
            {
                while (i < text.Length && char.IsWhiteSpace(text[i])) i++;
                if (i >= text.Length) break;
                var start = i;
                while (i < text.Length && !char.IsWhiteSpace(text[i])) i++;
                ids.Add((uint)ids.Count + 1);
                offsets.Add(new TokenOffset(start, i));
            }
            return new TokenizedSource(ids.ToArray(), offsets.ToArray());
        }

        public TokenizedModelInput EncodeModelInput(string text)
        {
            var source = TokenizeSource(text);
            var specialTokens = text.Length == 0 ? emptySpecialTokens : nonEmptySpecialTokens;
            var ids = source.Ids.Select(x => (long)x)
                .Concat(Enumerable.Repeat(0L, specialTokens))
                .ToArray();
            return new TokenizedModelInput(ids, Enumerable.Repeat(1L, ids.Length).ToArray(), null);
        }

        public int CountSourceTokens(string text) => TokenizeSource(text).Count;

        public int CountModelInputTokens(string text) =>
            CountSourceTokens(text) + (text.Length == 0 ? emptySpecialTokens : nonEmptySpecialTokens);

        public void Dispose() { }
    }
}
