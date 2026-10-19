# The measured notes, kept as a regression list

These are the reverse-engineering notes the emitter was built from: per
construct value counts, the shape-fetch groups, and the first single-fixture
verification. They are kept because they still describe what a *new* fixture
must reproduce, and because a change that breaks one of them is worth a look.

`TILEIR_ALIGNMENT.md` now states the rules by derivation. Where the two
disagree, the derivation is the intent and a difference here means a rule was
lost.

## Measured value counts, per construct

Every body op of every fixture has `Location::Unknown`, so a constant cannot be
attributed to its source position (`cutile/tests/ir_locate.rs`). The counts
below come from compiling one statement at a time and diffing the results
(`cutile/tests/ir_probe3.rs`, `ir_probe5.rs`, `ir_probe6.rs`, `ir_probe7.rs`,
`ir_probe8.rs`).

Most of the shape constants are **dead**: `Broadcast` and `Reshape` take their
shape as an attribute and name only one operand, so the constants their shape
arguments cost are printed but never consumed. cutile does not run DCE before
printing, and neither may this emitter.

| construct | values | breakdown |
| --- | --- | --- |
| `constant` `assume` `make_token` `make_tensor_view` `make_partition_view` `store_view_tko` arithmetic `reshape` `broadcast` | 1 | — |
| `get_tile_block_id` | 3 | plus 3 `assume`, so 6 in the body |
| `load_view_tko` | 2 | tile and token |
| `reduce` | 4 | `^bb0` args 2, body 1, result 1 |
| `a.broadcast(literal S)` over a scalar, rank `r` | 2r + (r+1) + 1 + r + r + 1 | `2r` target extents, `r+1` ones, `reshape`, `r` ones, `r` target, `broadcast` |
| `a.broadcast(x.shape())` over a scalar, rank `r` | 4r + (r+1) + 1 + r + r + 1 | `x.shape()` costs **4r**, twice what a literal shape costs |

The scalar broadcast is always three instructions — `reshape` the scalar to
all-ones, then `broadcast` — and the target shape is materialised twice before
the reshape and once more around the broadcast. Rank 1, rank 3 and rank 4 all
follow the same formula; it was checked at each rank.

## Verified: saxpy prints byte for byte

`scripts/check_tileir_alignment.sh` reports `saxpy PASS`. The emitter covers the
five constructs saxpy uses: a scalar broadcast, `load_like`, `load`, `store`, and
one binary arithmetic op.

## Verified: saxpy prints byte for byte

`scripts/check_tileir_alignment.sh` reports `saxpy PASS`. The emitter covers the
five constructs saxpy uses: a scalar broadcast, `load_like`, `load`, `store`, and
one binary arithmetic op.
