namespace OnnxTextEmbeddings;

internal sealed record PreparedChunk(
    string ModelInputText,
    string SourceText,
    string? Context,
    Utf16TextRange CharacterRange,
    TokenRange TokenRange,
    int SourceTokenCount,
    int SourceTokenCapacity,
    int DocumentTokenCount,
    ChunkBoundaryKind BoundaryKind,
    IReadOnlyList<string> HeadingPath,
    int ContextTokenCount,
    int SpecialTokenCount,
    TokenizedModelInput ModelInput);

internal interface ITextChunker
{
    IReadOnlyList<PreparedChunk> Chunk(string text, int modelInputLimit);
}

internal sealed class StructuredTextChunker(
    IEmbeddingTokenizer tokenizer,
    OnnxTextEmbeddingsOptions options) : ITextChunker
{
    private sealed record Section(int Start, int End, IReadOnlyList<string> HeadingPath);
    private sealed record Piece(int Start, int End, ChunkBoundaryKind BoundaryKind, IReadOnlyList<string> HeadingPath, bool Continuation, int Capacity);

    public IReadOnlyList<PreparedChunk> Chunk(string text, int modelInputLimit)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0)
            return Array.Empty<PreparedChunk>();

        var source = tokenizer.TokenizeSource(text);
        var specialTokens = Math.Max(0, tokenizer.CountModelInputTokens(string.Empty));
        var baseCapacity = modelInputLimit - specialTokens;
        if (baseCapacity <= 0)
            throw new ModelValidationException($"The configured input limit {modelInputLimit} leaves no room for source tokens.");

        if (source.Count <= baseCapacity)
            return new[] { FinalizePiece(text, source, new Piece(0, text.Length, ChunkBoundaryKind.WholeDocument, Array.Empty<string>(), false, baseCapacity), modelInputLimit, specialTokens) };

        var pieces = new List<Piece>();
        foreach (var section in ScanSections(text))
            SplitSection(text, source, section, baseCapacity, specialTokens, pieces);

        // Pack adjacent complete Markdown sections where possible. Split continuations remain isolated.
        var packed = new List<Piece>();
        foreach (var piece in pieces)
        {
            if (packed.Count > 0 && !piece.Continuation && !packed[^1].Continuation &&
                packed[^1].BoundaryKind == ChunkBoundaryKind.MarkdownSection && piece.BoundaryKind == ChunkBoundaryKind.MarkdownSection)
            {
                var previous = packed[^1];
                var combinedRange = new Utf16TextRange(previous.Start, piece.End - previous.Start);
                var combinedTokens = source.GetTokenRange(combinedRange).Length;
                if (combinedTokens <= baseCapacity)
                {
                    packed[^1] = previous with
                    {
                        End = piece.End,
                        HeadingPath = CommonPrefix(previous.HeadingPath, piece.HeadingPath),
                        Capacity = baseCapacity
                    };
                    continue;
                }
            }
            packed.Add(piece);
        }

        return packed.Select(piece => FinalizePiece(text, source, piece, modelInputLimit, specialTokens)).ToArray();
    }

    private void SplitSection(
        string text,
        TokenizedSource source,
        Section section,
        int baseCapacity,
        int specialTokens,
        List<Piece> output)
    {
        var fullRange = new Utf16TextRange(section.Start, section.End - section.Start);
        var sectionTokenRange = source.GetTokenRange(fullRange);
        if (sectionTokenRange.Length <= baseCapacity)
        {
            output.Add(new Piece(section.Start, section.End, ChunkBoundaryKind.MarkdownSection, section.HeadingPath, false, baseCapacity));
            return;
        }

        var context = options.Chunking.RepeatHeadingContext && section.HeadingPath.Count > 0
            ? string.Join(" > ", section.HeadingPath)
            : null;
        var contextTokens = string.IsNullOrEmpty(context) ? 0 : tokenizer.CountSourceTokens(context + "\n\n");
        var continuationCapacity = Math.Max(1, baseCapacity - contextTokens);

        var cursor = section.Start;
        var first = true;
        while (cursor < section.End)
        {
            var remainingRange = new Utf16TextRange(cursor, section.End - cursor);
            var remainingTokens = source.GetTokenRange(remainingRange);
            var capacity = first ? baseCapacity : continuationCapacity;
            if (remainingTokens.Length <= capacity)
            {
                output.Add(new Piece(cursor, section.End, first ? ChunkBoundaryKind.MarkdownSection : ChunkBoundaryKind.ParagraphGroup, section.HeadingPath, !first, capacity));
                break;
            }

            var tokenStart = remainingTokens.Start;
            var targetEnd = source.CharacterEndForTokenCount(tokenStart, capacity, section.End);
            targetEnd = Math.Clamp(targetEnd, cursor + 1, section.End);
            var (breakAt, kind) = FindPreferredBreak(text, cursor, targetEnd);
            if (breakAt <= cursor)
            {
                breakAt = targetEnd;
                kind = ChunkBoundaryKind.TokenWindow;
            }
            output.Add(new Piece(cursor, breakAt, kind, section.HeadingPath, !first, capacity));
            cursor = breakAt;
            first = false;
        }
    }

    private PreparedChunk FinalizePiece(
        string document,
        TokenizedSource source,
        Piece piece,
        int modelInputLimit,
        int specialTokens)
    {
        var sourceText = document.Substring(piece.Start, piece.End - piece.Start);
        var context = piece.Continuation && options.Chunking.RepeatHeadingContext && piece.HeadingPath.Count > 0
            ? string.Join(" > ", piece.HeadingPath)
            : null;
        var modelText = string.IsNullOrEmpty(context) ? sourceText : context + "\n\n" + sourceText;
        var encoded = tokenizer.EncodeModelInput(modelText);
        if (encoded.TokenCount > modelInputLimit)
            throw new ModelValidationException($"Chunk finalization produced {encoded.TokenCount} model tokens, exceeding the configured limit of {modelInputLimit}. Reduce the document chunk limit or heading context.");

        var characterRange = new Utf16TextRange(piece.Start, piece.End - piece.Start);
        var tokenRange = source.GetTokenRange(characterRange);
        var contextTokens = string.IsNullOrEmpty(context) ? 0 : tokenizer.CountSourceTokens(context + "\n\n");
        return new PreparedChunk(
            modelText,
            sourceText,
            context,
            characterRange,
            tokenRange,
            tokenRange.Length,
            piece.Capacity,
            source.Count,
            piece.BoundaryKind,
            piece.HeadingPath,
            contextTokens,
            specialTokens,
            encoded);
    }

    private static (int BreakAt, ChunkBoundaryKind Kind) FindPreferredBreak(string text, int start, int targetEnd)
    {
        // paragraph boundary
        for (var i = targetEnd - 1; i > start; i--)
        {
            if (i + 1 < text.Length && text[i] == '\n' && text[i + 1] == '\n')
                return (i + 2, ChunkBoundaryKind.ParagraphGroup);
            if (i + 3 < text.Length && text.AsSpan(i, 4).SequenceEqual("\r\n\r\n"))
                return (i + 4, ChunkBoundaryKind.ParagraphGroup);
        }

        // sentence boundary
        for (var i = targetEnd - 1; i > start; i--)
        {
            if (text[i] is '.' or '!' or '?' or '。' or '！' or '？')
                return (i + 1, ChunkBoundaryKind.SentenceGroup);
        }

        // word boundary
        for (var i = targetEnd - 1; i > start; i--)
        {
            if (char.IsWhiteSpace(text[i]))
                return (i + 1, ChunkBoundaryKind.WordGroup);
        }

        return (targetEnd, ChunkBoundaryKind.TokenWindow);
    }

    private static IReadOnlyList<Section> ScanSections(string text)
    {
        var headings = new List<(int Start, int Level, string Title, IReadOnlyList<string> Path)>();
        var stack = new string?[6];
        var inFence = false;
        var lineStart = 0;
        while (lineStart < text.Length)
        {
            var newline = text.IndexOf('\n', lineStart);
            var lineEnd = newline < 0 ? text.Length : newline;
            var line = text.AsSpan(lineStart, lineEnd - lineStart).TrimEnd('\r');
            var trimmed = line.TrimStart();
            if (trimmed.StartsWith("```", StringComparison.Ordinal) || trimmed.StartsWith("~~~", StringComparison.Ordinal))
                inFence = !inFence;
            else if (!inFence)
            {
                var hashes = 0;
                while (hashes < trimmed.Length && hashes < 6 && trimmed[hashes] == '#') hashes++;
                if (hashes > 0 && hashes < trimmed.Length && char.IsWhiteSpace(trimmed[hashes]))
                {
                    var title = trimmed[hashes..].Trim().ToString().TrimEnd('#').Trim();
                    if (title.Length > 0)
                    {
                        stack[hashes - 1] = title;
                        for (var i = hashes; i < stack.Length; i++) stack[i] = null;
                        headings.Add((lineStart, hashes, title, stack.Take(hashes).Where(x => x is not null).Select(x => x!).ToArray()));
                    }
                }
            }
            if (newline < 0) break;
            lineStart = newline + 1;
        }

        if (headings.Count == 0)
            return new[] { new Section(0, text.Length, Array.Empty<string>()) };

        var sections = new List<Section>();
        if (headings[0].Start > 0)
            sections.Add(new Section(0, headings[0].Start, Array.Empty<string>()));
        for (var i = 0; i < headings.Count; i++)
        {
            var end = i + 1 < headings.Count ? headings[i + 1].Start : text.Length;
            sections.Add(new Section(headings[i].Start, end, headings[i].Path));
        }
        return sections;
    }

    private static IReadOnlyList<string> CommonPrefix(IReadOnlyList<string> left, IReadOnlyList<string> right)
    {
        var count = Math.Min(left.Count, right.Count);
        var result = new List<string>(count);
        for (var i = 0; i < count && left[i] == right[i]; i++)
            result.Add(left[i]);
        return result;
    }
}
