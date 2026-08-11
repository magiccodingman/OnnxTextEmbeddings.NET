namespace OnnxTextEmbeddings;

/// <summary>Helpers for mathematically combining chunk embeddings into one lossy semantic representation.</summary>
public static class TextEmbeddingAggregationExtensions
{
    private const double AffinityExponent = 4.0;
    private const double RedundancyExponent = 0.5;
    private const double DegenerateCoherenceEpsilon = 1e-3;

    /// <summary>
    /// Combines compatible document embeddings into exactly one semantic vector. Multi-vector retrieval remains the
    /// preferred representation when it is available; this operation is intentionally lossy semantic compression.
    /// </summary>
    public static SingleEmbedding CombineToSingle(
        this IReadOnlyList<TextEmbedding> embeddings,
        SingleEmbeddingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        if (embeddings.Count == 0)
            throw new ArgumentException("At least one embedding is required.", nameof(embeddings));

        options ??= new SingleEmbeddingOptions();
        if (options.AggregationStrategy != EmbeddingAggregationStrategy.SemanticCoverage)
            throw new ArgumentOutOfRangeException(nameof(options), "Unsupported aggregation strategy.");
        if (options.OutputFormat is EmbeddingVectorFormat.Unspecified)
            throw new ArgumentOutOfRangeException(nameof(options), "OutputFormat cannot be Unspecified.");

        var first = embeddings[0] ?? throw new ArgumentException("Embedding collections cannot contain null records.", nameof(embeddings));
        var sourceDimensions = first.Vector.Dimensions;
        var fingerprint = first.Identity.EmbeddingSpaceFingerprint;
        if (sourceDimensions <= 0)
            throw new ArgumentException("Embedding dimensions must be positive.", nameof(embeddings));

        for (var i = 0; i < embeddings.Count; i++)
        {
            var embedding = embeddings[i] ?? throw new ArgumentException("Embedding collections cannot contain null records.", nameof(embeddings));
            if (embedding.Vector.Dimensions != sourceDimensions)
                throw new ArgumentException(
                    $"Embedding dimensions differ: expected {sourceDimensions}, found {embedding.Vector.Dimensions} at index {i}.",
                    nameof(embeddings));
            if (!fingerprint.Equals(embedding.Identity.EmbeddingSpaceFingerprint, StringComparison.Ordinal))
                throw new EmbeddingSpaceMismatchException(
                    $"Embedding at index {i} belongs to space '{embedding.Identity.EmbeddingSpaceFingerprint}', expected '{fingerprint}'.");
        }

        var outputDimensions = options.OutputDimensions ?? sourceDimensions;
        if (outputDimensions <= 0 || outputDimensions > sourceDimensions)
            throw new ArgumentOutOfRangeException(
                nameof(options),
                $"Requested embedding dimension {outputDimensions} exceeds the maximum available dimension of {sourceDimensions}.");

        var neutralBaseline = options.NeutralSimilarityBaseline ?? 0f;
        if (!float.IsFinite(neutralBaseline) || neutralBaseline < -1f || neutralBaseline >= 1f)
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "NeutralSimilarityBaseline must be finite and in the range [-1, 1).");

        var targetFormat = ResolveOutputFormat(embeddings, options.OutputFormat);
        var mass = CalculateSourceMass(embeddings);

        if (embeddings.Count == 1)
            return CombineSingle(first, sourceDimensions, outputDimensions, targetFormat, mass, options);

        var unitVectors = new float[embeddings.Count][];
        for (var i = 0; i < embeddings.Count; i++)
        {
            unitVectors[i] = embeddings[i].Vector.ToFloat32();
            EmbeddingVectorMath.NormalizeInPlace(unitVectors[i]);
        }

        var densities = mass.PerEmbeddingMass.Select(value => (double)value).ToArray();
        for (var i = 0; i < unitVectors.Length; i++)
        {
            for (var j = i + 1; j < unitVectors.Length; j++)
            {
                var similarity = Dot(unitVectors[i], unitVectors[j]);
                var recentered = Math.Clamp((similarity - neutralBaseline) / (1.0 - neutralBaseline), 0.0, 1.0);
                var affinity = Math.Pow(recentered, AffinityExponent);
                densities[i] += mass.PerEmbeddingMass[j] * affinity;
                densities[j] += mass.PerEmbeddingMass[i] * affinity;
            }
        }

        var weights = new double[embeddings.Count];
        var weightSum = 0.0;
        for (var i = 0; i < weights.Length; i++)
        {
            var sourceMass = mass.PerEmbeddingMass[i];
            if (sourceMass <= 0)
                continue;
            if (!(densities[i] > 0) || !double.IsFinite(densities[i]))
                throw new InvalidOperationException("Semantic redundancy density became invalid.");
            weights[i] = sourceMass / Math.Pow(densities[i], RedundancyExponent);
            weightSum += weights[i];
        }
        if (!(weightSum > 0) || !double.IsFinite(weightSum))
            throw new InvalidOperationException("Source-content mass produced no usable aggregation weight.");

        var weighted = new double[sourceDimensions];
        for (var i = 0; i < unitVectors.Length; i++)
        {
            if (weights[i] == 0) continue;
            var vector = unitVectors[i];
            for (var d = 0; d < vector.Length; d++)
                weighted[d] += weights[i] * vector[d];
        }

        var weightedNorm = Norm(weighted);
        var coherence = weightedNorm / weightSum;
        var fallbackUsed = coherence < DegenerateCoherenceEpsilon;
        float[] aggregate;
        if (fallbackUsed)
        {
            aggregate = unitVectors[FindWeightedMedoid(unitVectors, weights)].ToArray();
        }
        else
        {
            aggregate = new float[sourceDimensions];
            var inverseNorm = 1.0 / weightedNorm;
            for (var d = 0; d < aggregate.Length; d++)
                aggregate[d] = (float)(weighted[d] * inverseNorm);
        }
        EmbeddingVectorMath.NormalizeInPlace(aggregate);

        var minimumSourceSimilarity = float.PositiveInfinity;
        foreach (var vector in unitVectors)
            minimumSourceSimilarity = Math.Min(minimumSourceSimilarity, (float)Dot(aggregate, vector));

        var identity = first.Identity with { IsNormalized = true };
        EmbeddingDimensionReductionInfo? reduction = null;
        var outputValues = aggregate;
        if (outputDimensions < sourceDimensions)
        {
            var profile = EmbeddingDimensionReduction.ResolveProfile(options.DimensionReductionStrategy);
            outputValues = EmbeddingDimensionReduction.Reduce(aggregate, outputDimensions, options.DimensionReductionStrategy);
            identity = EmbeddingDimensionReduction.CreateReducedIdentity(identity, sourceDimensions, outputDimensions, profile);
            reduction = new EmbeddingDimensionReductionInfo
            {
                ProfileId = profile,
                ProfileVersion = 1,
                SourceDimensions = sourceDimensions,
                OutputDimensions = outputDimensions
            };
        }

        return new SingleEmbedding
        {
            Vector = EmbeddingVector.FromFloat32(outputValues, targetFormat),
            Identity = identity,
            RepresentationKind = EmbeddingRepresentationKind.Aggregated,
            SourceEmbeddingCount = embeddings.Count,
            SourceTokenCount = mass.TotalSourceTokens,
            SourceDimensions = sourceDimensions,
            Aggregation = new EmbeddingAggregationInfo
            {
                ProfileId = EmbeddingAggregationProfiles.SemanticCoverageV1,
                ProfileVersion = 1,
                SourceMassMethod = mass.Method,
                NeutralSimilarityBaseline = neutralBaseline,
                AffinityExponent = (float)AffinityExponent,
                RedundancyExponent = (float)RedundancyExponent,
                AggregationCoherence = (float)Math.Clamp(coherence, 0.0, 1.0),
                MinimumSourceSimilarity = minimumSourceSimilarity,
                FallbackUsed = fallbackUsed
            },
            DimensionReduction = reduction
        };
    }

    private static SingleEmbedding CombineSingle(
        TextEmbedding embedding,
        int sourceDimensions,
        int outputDimensions,
        EmbeddingVectorFormat targetFormat,
        SourceMassResult mass,
        SingleEmbeddingOptions options)
    {
        var noDimensionChange = outputDimensions == sourceDimensions;
        var noFormatChange = targetFormat == embedding.Vector.Format;
        var vector = embedding.Vector;
        var identity = embedding.Identity;
        EmbeddingDimensionReductionInfo? reduction = null;

        if (!noDimensionChange || !noFormatChange)
        {
            var values = embedding.Vector.ToFloat32();
            EmbeddingVectorMath.NormalizeInPlace(values);
            if (!noDimensionChange)
            {
                var profile = EmbeddingDimensionReduction.ResolveProfile(options.DimensionReductionStrategy);
                values = EmbeddingDimensionReduction.Reduce(values, outputDimensions, options.DimensionReductionStrategy);
                identity = EmbeddingDimensionReduction.CreateReducedIdentity(identity, sourceDimensions, outputDimensions, profile);
                reduction = new EmbeddingDimensionReductionInfo
                {
                    ProfileId = profile,
                    ProfileVersion = 1,
                    SourceDimensions = sourceDimensions,
                    OutputDimensions = outputDimensions
                };
            }
            else
            {
                identity = identity with { IsNormalized = true };
            }
            vector = EmbeddingVector.FromFloat32(values, targetFormat);
        }

        return new SingleEmbedding
        {
            Vector = vector,
            Identity = identity,
            RepresentationKind = EmbeddingRepresentationKind.Direct,
            SourceEmbeddingCount = 1,
            SourceTokenCount = mass.TotalSourceTokens,
            SourceDimensions = sourceDimensions,
            Aggregation = new EmbeddingAggregationInfo
            {
                ProfileId = EmbeddingAggregationProfiles.Passthrough,
                ProfileVersion = 1,
                SourceMassMethod = mass.Method,
                NeutralSimilarityBaseline = options.NeutralSimilarityBaseline ?? 0f,
                AffinityExponent = (float)AffinityExponent,
                RedundancyExponent = (float)RedundancyExponent,
                AggregationCoherence = 1f,
                MinimumSourceSimilarity = 1f,
                FallbackUsed = false
            },
            DimensionReduction = reduction
        };
    }

    private static EmbeddingVectorFormat ResolveOutputFormat(
        IReadOnlyList<TextEmbedding> embeddings,
        EmbeddingVectorFormat? requested)
    {
        if (requested is { } explicitFormat)
            return explicitFormat;
        var common = embeddings[0].Vector.Format;
        for (var i = 1; i < embeddings.Count; i++)
        {
            if (embeddings[i].Vector.Format != common)
                return EmbeddingVectorFormat.Float32;
        }
        return common;
    }

    private static SourceMassResult CalculateSourceMass(IReadOnlyList<TextEmbedding> embeddings)
    {
        var commonDocumentTokenCount = embeddings[0].Source.DocumentTokenCount;
        var canUseRanges = commonDocumentTokenCount > 0;
        for (var i = 0; i < embeddings.Count; i++)
        {
            var source = embeddings[i].Source;
            if (source.TokenCount < 0)
                throw new ArgumentException($"Embedding at index {i} has a negative source token count.", nameof(embeddings));
            if (source.DocumentTokenCount != commonDocumentTokenCount)
                canUseRanges = false;
            if (source.TokenRange.Start < 0 || source.TokenRange.Length <= 0)
                canUseRanges = false;
            else if (commonDocumentTokenCount > 0 && source.TokenRange.End > commonDocumentTokenCount)
                canUseRanges = false;
        }

        if (!canUseRanges)
        {
            var fallback = embeddings.Select(x => (double)x.Source.TokenCount).ToArray();
            var total = fallback.Sum();
            if (!(total > 0))
                throw new ArgumentException("Embeddings must carry positive source-content mass.", nameof(embeddings));
            return new SourceMassResult(
                fallback,
                checked((int)Math.Round(total, MidpointRounding.AwayFromZero)),
                EmbeddingSourceMassMethod.SourceTokenCount);
        }

        var boundaries = embeddings
            .SelectMany(x => new[] { x.Source.TokenRange.Start, x.Source.TokenRange.End })
            .Distinct()
            .OrderBy(x => x)
            .ToArray();
        var mass = new double[embeddings.Count];
        var totalUnique = 0;
        for (var boundary = 0; boundary < boundaries.Length - 1; boundary++)
        {
            var start = boundaries[boundary];
            var end = boundaries[boundary + 1];
            var length = end - start;
            if (length <= 0) continue;

            var covering = new List<int>();
            for (var i = 0; i < embeddings.Count; i++)
            {
                var range = embeddings[i].Source.TokenRange;
                if (range.Start <= start && range.End >= end)
                    covering.Add(i);
            }
            if (covering.Count == 0) continue;

            totalUnique += length;
            var share = (double)length / covering.Count;
            foreach (var index in covering)
                mass[index] += share;
        }

        if (totalUnique <= 0 || mass.All(x => x <= 0))
            throw new ArgumentException("Embedding token ranges provide no usable source-content mass.", nameof(embeddings));
        return new SourceMassResult(mass, totalUnique, EmbeddingSourceMassMethod.TokenRangeCoverage);
    }

    private static int FindWeightedMedoid(IReadOnlyList<float[]> vectors, IReadOnlyList<double> weights)
    {
        var bestIndex = 0;
        var bestScore = double.NegativeInfinity;
        for (var i = 0; i < vectors.Count; i++)
        {
            double score = 0;
            for (var j = 0; j < vectors.Count; j++)
                score += weights[j] * Dot(vectors[i], vectors[j]);
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private static double Dot(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
    {
        double result = 0;
        for (var i = 0; i < left.Length; i++)
            result += (double)left[i] * right[i];
        return result;
    }

    private static double Norm(ReadOnlySpan<double> values)
    {
        double sum = 0;
        foreach (var value in values)
            sum += value * value;
        return Math.Sqrt(sum);
    }

    private sealed record SourceMassResult(
        double[] PerEmbeddingMass,
        int TotalSourceTokens,
        EmbeddingSourceMassMethod Method);
}
