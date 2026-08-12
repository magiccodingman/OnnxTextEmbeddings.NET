using System.Text;
using Tokenizers.HuggingFace.Tokenizer;

namespace OnnxTextEmbeddings;

internal readonly record struct TokenOffset(int Start, int End);

internal sealed record TokenizedSource(
    uint[] Ids,
    TokenOffset[] Offsets)
{
    public int Count => Ids.Length;

    public TokenRange GetTokenRange(Utf16TextRange range)
    {
        if (range.Length == 0 || Offsets.Length == 0)
            return new TokenRange(0, 0);
        var first = 0;
        while (first < Offsets.Length && Offsets[first].End <= range.Start)
            first++;
        var last = first;
        while (last < Offsets.Length && Offsets[last].Start < range.End)
            last++;
        return new TokenRange(first, Math.Max(0, last - first));
    }

    public int CharacterEndForTokenCount(int tokenStart, int tokenCount, int fallbackEnd)
    {
        if (tokenCount <= 0 || tokenStart >= Offsets.Length)
            return fallbackEnd;
        var index = Math.Min(Offsets.Length - 1, tokenStart + tokenCount - 1);
        return Offsets[index].End;
    }
}

internal sealed record TokenizedModelInput(
    long[] InputIds,
    long[] AttentionMask,
    long[]? TokenTypeIds)
{
    public int TokenCount => InputIds.Length;
}

internal interface IEmbeddingTokenizer : IDisposable
{
    TokenizedSource TokenizeSource(string text);
    TokenizedModelInput EncodeModelInput(string text);
    int CountSourceTokens(string text);
    int CountModelInputTokens(string text);
}

internal sealed class HuggingFaceEmbeddingTokenizer : IEmbeddingTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly object _sync = new();

    public HuggingFaceEmbeddingTokenizer(string tokenizerPath)
    {
        try
        {
            _tokenizer = Tokenizer.FromFile(tokenizerPath);
        }
        catch (Exception ex)
        {
            throw new TokenizerCompatibilityException($"Unable to load Hugging Face tokenizer '{tokenizerPath}'.", ex);
        }
    }

    public TokenizedSource TokenizeSource(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        lock (_sync)
        {
            var encoding = _tokenizer.Encode(
                text,
                addSpecialTokens: false,
                includeOffsets: true,
                charOffsets: false).First();
            var byteToUtf16 = BuildUtf8ByteToUtf16Map(text);
            var offsets = encoding.Offsets.Select(offset => new TokenOffset(
                MapByteOffset(byteToUtf16, checked((int)offset.Start)),
                MapByteOffset(byteToUtf16, checked((int)offset.End)))).ToArray();
            return new TokenizedSource(encoding.Ids.ToArray(), offsets);
        }
    }

    public TokenizedModelInput EncodeModelInput(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        lock (_sync)
        {
            var encoding = _tokenizer.Encode(
                text,
                addSpecialTokens: true,
                includeTypeIds: true,
                includeAttentionMask: true).First();
            return new TokenizedModelInput(
                encoding.Ids.Select(x => (long)x).ToArray(),
                encoding.AttentionMask.Count == 0
                    ? Enumerable.Repeat(1L, encoding.Ids.Count).ToArray()
                    : encoding.AttentionMask.Select(x => (long)x).ToArray(),
                encoding.TypeIds.Count == 0 ? null : encoding.TypeIds.Select(x => (long)x).ToArray());
        }
    }

    public int CountSourceTokens(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        lock (_sync)
            return _tokenizer.Encode(text, addSpecialTokens: false).First().Ids.Count;
    }

    public int CountModelInputTokens(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        lock (_sync)
            return _tokenizer.Encode(text, addSpecialTokens: true).First().Ids.Count;
    }

    public void Dispose() => _tokenizer.Dispose();

    private static int[] BuildUtf8ByteToUtf16Map(string text)
    {
        var byteCount = System.Text.Encoding.UTF8.GetByteCount(text);
        var map = new int[byteCount + 1];
        var utf16 = 0;
        var utf8 = 0;
        while (utf16 < text.Length)
        {
            var status = Rune.DecodeFromUtf16(text.AsSpan(utf16), out var rune, out var consumed);
            if (status != System.Buffers.OperationStatus.Done)
                throw new TokenizerCompatibilityException("Input contains invalid UTF-16 data.");
            var bytes = rune.Utf8SequenceLength;
            for (var i = 0; i < bytes; i++)
                map[utf8 + i] = utf16;
            utf8 += bytes;
            utf16 += consumed;
            map[utf8] = utf16;
        }
        return map;
    }

    private static int MapByteOffset(int[] map, int byteOffset)
    {
        if (byteOffset < 0 || byteOffset >= map.Length)
            throw new TokenizerCompatibilityException($"Tokenizer returned invalid byte offset {byteOffset}.");
        return map[byteOffset];
    }
}
