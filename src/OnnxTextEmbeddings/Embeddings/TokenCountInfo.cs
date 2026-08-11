namespace OnnxTextEmbeddings;

/// <summary>Non-throwing query token-count information for validation before embedding.</summary>
public sealed record QueryTokenCount(
    int SourceTokenCount,
    int InputTokenCount,
    int QueryMaxTokens,
    int? ModelMaxTokens)
{
    public bool FitsConfiguredLimit => InputTokenCount <= QueryMaxTokens;
    public bool FitsModelLimit => ModelMaxTokens is null || InputTokenCount <= ModelMaxTokens.Value;
    public bool Fits => FitsConfiguredLimit && FitsModelLimit;
}
