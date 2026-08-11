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
        {
            var whole = new Piece(0, text.Length, ChunkBoundaryKind.WholeDocument, Array.Empty<string>(), false, baseCapacity);
            return FinalizePieces(text, source, new[] { whole }, modelInputLimit, specialTokens);
        }

        var pieces = new List<Piece>();
        foreach (var section in ScanSections(text))
            SplitSection(text, source, section, baseCapacity, pieces);

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

        return FinalizePieces(text, source, packed, modelInputLimit, specialTokens);
    }

    private void SplitSection(
        string text,
        TokenizedSource source,
        Section section,
        int baseCapacity,
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
        var overlapTokens = Math.Min(options.Chunking.ChunkOverlapTokens, Math.Max(0, continuationCapacity - 1));

        var cursor = section.Start;
        var first = true;
        while (cursor < section.End)
        {
            var remainingRange = new Utf16TextRange(cursor, section.End - cursor);
            var remainingTokens = source.GetTokenRange(remainingRange);
            var capacity = first ? baseCapacity : continuationCapacity;
            var overlap = first ? 0 : overlapTokens;
            var newContentCapacity = Math.Max(1, capacity - overlap);
            var pieceStart = first ? cursor : FindOverlapStart(source, section.Start, cursor, overlap);

            if (remainingTokens.Length <= newContentCapacity)
            {
                output.Add(new Piece(
                    pieceStart,
                    section.End,
                    first ? ChunkBoundaryKind.MarkdownSection : ChunkBoundaryKind.ParagraphGroup,
                    section.HeadingPath,
                    !first,
                    capacity));
                break;
            }

            var tokenStart = remainingTokens.Start;
            var targetEnd = source.CharacterEndForTokenCount(tokenStart, newContentCapacity, section.End);
            targetEnd = Math.Clamp(targetEnd, cursor + 1, section.End);
            var (breakAt, kind) = FindPreferredBreak(text, cursor, targetEnd);
            if (breakAt <= cursor)
            {
                breakAt = targetEnd;
                kind = ChunkBoundaryKind.TokenWindow;
            }

            output.Add(new Piece(pieceStart, breakAt, kind, section.HeadingPath, !first, capacity));
            cursor = breakAt;
            first = false;
        }
    }

    private static int FindOverlapStart(
        TokenizedSource source,
        int sectionStart,
        int cursor,
        int overlapTokens)
    {
        if (overlapTokens <= 0 || cursor <= sectionStart)
            return cursor;

        var priorRange = source.GetTokenRange(new Utf16TextRange(sectionStart, cursor - sectionStart));
        if (priorRange.Length <= 0)
            return cursor;

        var overlapStartToken = Math.Max(priorRange.Start, priorRange.End - overlapTokens);
        if (overlapStartToken < 0 || overlapStartToken >= source.Offsets.Length)
            return cursor;

        return Math.Max(sectionStart, source.Offsets[overlapStartToken].Start);
    }

    private IReadOnlyList<PreparedChunk> FinalizePieces(
        string document,
        TokenizedSource source,
        IReadOnlyList<Piece> pieces,
        int modelInputLimit,
        int specialTokens)
    {
        var output = new List<PreparedChunk>(pieces.Count);
        foreach (var piece in pieces)
            FinalizePieceWithExactLimit(document, source, piece, modelInputLimit, specialTokens, output);
        return output;
    }

    private void FinalizePieceWithExactLimit(
        string document,
        TokenizedSource source,
        Piece piece,
        int modelInputLimit,
        int specialTokens,
        List<PreparedChunk> output)
    {
        var remaining = piece;
        while (remaining.Start < remaining.End)
        {
            if (TryFinalizePiece(document, source, remaining, modelInputLimit, specialTokens, out var finalized))
            {
                output.Add(finalized!);
                return;
            }

            var fittingEnd = FindLargestFittingEnd(document, source, remaining, modelInputLimit);
            if (fittingEnd <= remaining.Start || fittingEnd >= remaining.End)
            {
                throw new ModelValidationException(
                    $"The configured input limit {modelInputLimit} is too small to finalize the next source token with its required model/context tokens.");
            }

            var (preferredEnd, preferredKind) = FindPreferredBreak(document, remaining.Start, fittingEnd);
            var first = preferredEnd > remaining.Start
                ? remaining with { End = preferredEnd, BoundaryKind = preferredKind }
                : remaining with { End = fittingEnd, BoundaryKind = ChunkBoundaryKind.TokenWindow };

            if (!TryFinalizePiece(document, source, first, modelInputLimit, specialTokens, out finalized))
            {
                first = remaining with { End = fittingEnd, BoundaryKind = ChunkBoundaryKind.TokenWindow };
                if (!TryFinalizePiece(document, source, first, modelInputLimit, specialTokens, out finalized))
                    throw new ModelValidationException("Unable to finalize a corrective chunk split within the configured token limit.");
            }

            output.Add(finalized!);
            remaining = CreateCorrectiveRemainder(first.End, remaining, modelInputLimit, specialTokens);
        }
    }

    private int FindLargestFittingEnd(
        string document,
        TokenizedSource source,
        Piece piece,
        int modelInputLimit)
    {
        var range = source.GetTokenRange(new Utf16TextRange(piece.Start, piece.End - piece.Start));
        for (var tokenCount = range.Length - 1; tokenCount >= 1; tokenCount--)
        {
            var candidateEnd = source.CharacterEndForTokenCount(range.Start, tokenCount, piece.End);
            if (candidateEnd <= piece.Start || candidateEnd >= piece.End)
                continue;

            var candidate = piece with { End = candidateEnd };
            if (CountModelInputTokens(document, candidate) <= modelInputLimit)
                return candidateEnd;
        }
        return -1;
    }

    private Piece CreateCorrectiveRemainder(
        int start,
        Piece original,
        int modelInputLimit,
        int specialTokens)
    {
        var context = options.Chunking.RepeatHeadingContext && original.HeadingPath.Count > 0
            ? string.Join(" > ", original.HeadingPath)
            : null;
        var contextTokens = string.IsNullOrEmpty(context) ? 0 : tokenizer.CountSourceTokens(context + "\n\n");
        var capacity = Math.Max(1, modelInputLimit - specialTokens - contextTokens);
        return new Piece(
            start,
            original.End,
            ChunkBoundaryKind.TokenWindow,
            original.HeadingPath,
            true,
            capacity);
    }

    private int CountModelInputTokens(string document, Piece piece)
    {
        var sourceText = document.Substring(piece.Start, piece.End - piece.Start);
        var context = piece.Continuation && options.Chunking.RepeatHeadingContext && piece.HeadingPath.Count > 0
            ? string.Join(" > ", piece.HeadingPath)
            : null;
        var modelText = string.IsNullOrEmpty(context) ? sourceText : context + "\n\n" + sourceText;
        return tokenizer.EncodeModelInput(modelText).TokenCount;
    }

    private bool TryFinalizePiece(
        string document,
        TokenizedSource source,
        Piece piece,
        int modelInputLimit,
        int specialTokens,
        out PreparedChunk? chunk)
    {
        var sourceText = document.Substring(piece.Start, piece.End - piece.Start);
        var context = piece.Continuation && options.Chunking.RepeatHeadingContext && piece.HeadingPath.Count > 0
            ? string.Join(" > ", piece.HeadingPath)
            : null;
        var modelText = string.IsNullOrEmpty(context) ? sourceText : context + "\n\n" + sourceText;
        var encoded = tokenizer.EncodeModelInput(modelText);
        if (encoded.TokenCount > modelInputLimit)
        {
            chunk = null;
            return false;
        }

        var characterRange = new Utf16TextRange(piece.Start, piece.End - piece.Start);
        var tokenRange = source.GetTokenRange(characterRange);
        var contextTokens = string.IsNullOrEmpty(context) ? 0 : tokenizer.CountSourceTokens(context + "\n\n");
        chunk = new PreparedChunk(
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
        return true;
    }

    private static (int BreakAt, ChunkBoundaryKind Kind) FindPreferredBreak(string text, int start, int targetEnd)
    {
        for (var i = targetEnd - 1; i > start; i--)
        {
            if (i + 1 < text.Length && text[i] == '\n' && text[i + 1] == '\n')
                return (i + 2, ChunkBoundaryKind.ParagraphGroup);
            if (i + 3 < text.Length && text.AsSpan(i, 4).SequenceEqual("\r\n\r\n"))
                return (i + 4, ChunkBoundaryKind.ParagraphGroup);
        }

        for (var i = targetEnd - 1; i > start; i--)
        {
            if (text[i] is '.' or '!' or '?' or '。' or '！' or '？')
                return (i + 1, ChunkBoundaryKind.SentenceGroup);
        }

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
