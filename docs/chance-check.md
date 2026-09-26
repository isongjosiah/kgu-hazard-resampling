# The chance check

Code: `src/hazres/grid/realise.py` (uses the Rust engine's zone-mean functions).

A difference between conversion methods only counts if it is bigger than the
differences between versions of the data that the coarse layer cannot tell
apart. So for each coarse layer we make about 10 **random but equally
believable fine-scale versions**. Each one:

1. starts from a smooth base that already gives every coarse cell its published
   average (bilinear, or downscaled when terrain is given);
2. adds random fine-scale detail with a chosen strength (`sd`) and grain
   (`correlation_m`), with its average removed inside every coarse cell and its
   spread set exactly to `sd`;
3. is shifted once more so every coarse cell's average is exactly the published
   value.

Every version therefore gives back exactly the downloaded data when averaged
over the coarse cells. They differ only in detail the coarse data cannot see.

## The assumption: how much detail

The strength of the detail cannot be learned from the coarse layer. For a rough
layer, averaging to 1 km destroys the fine detail, and the coarse numbers do not
say how much there was. On the synthetic study, where the truth is known:

| Synthetic layer | True detail inside a coarse cell | Automatic estimate |
|---|---|---|
| Rainfall (smooth) | 0.11 | 0.19 (too high) |
| Bedrock depth (rough) | 0.89 | 0.43 (too low) |

The automatic estimate (the spread between neighbouring coarse cells, divided
by √2; grain half a coarse cell) is therefore only a transparent default. The
strength used is always reported, and the check is always run at 0.5, 1 and 2
times it. Given the true strength, the generator reproduces the truth's detail
within 10% (tested).

## An early warning for the pilot

Comparing the converted *fields* themselves on the synthetic study (average
root-mean-square difference between pairs):

| Layer | Between the methods | Between random versions (x0.5 / x1 / x2) |
|---|---|---|
| Rainfall (smooth) | 0.15 | 0.14 / 0.28 / 0.55 |
| Bedrock depth (rough) | 0.14 | 0.31 / 0.61 / 1.22 |

At the level of the input values, the methods differ *less* than random
versions do. This does not decide the question: the test is on what the
*models* do (which factors they rely on, which places they rank highest), and
the methods differ systematically (blocky versus smooth) while the random
versions differ by noise, which models may treat very differently. But it means
the verdict may depend strongly on the assumed amount of detail, and the pilot
must look at this first.

## Not yet

- Class layers (rock type): realisations are not built yet.
