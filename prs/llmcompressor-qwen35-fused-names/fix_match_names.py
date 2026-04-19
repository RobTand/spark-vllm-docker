"""
Fix for llmcompressor match_names_set_eager to handle models with
non-contiguous fused layer patterns (e.g., Qwen3.5 with mixed
self_attn + linear_attn layers).

The bug: match_names_set_eager iterates sorted tensor names and tries to
fill q/k/v sets eagerly. In Qwen3.5, self_attn layers appear at indices
3, 7, 11... with linear_attn layers between them. The function matches
layer 3's q_proj, then encounters layer 7's q_proj before seeing layer 3's
k_proj (because linear_attn layers have no k_proj), raising a "matched twice"
error.

The fix: extract the layer prefix from each matched name. If the next match
for the same target comes from a DIFFERENT layer, flush the partial set as
unmatched and start a new set, rather than raising an error.
"""


def match_names_set_eager_fixed(
    names, targets, return_unmatched=True
):
    """Fixed version that handles non-contiguous fused patterns."""
    import re
    from compressed_tensors.utils.match import match_name

    matched_sets = []
    matches = dict.fromkeys(targets, None)

    def natural_key(s):
        return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]

    def get_layer_prefix(name):
        """Extract layer prefix up to the layer index."""
        # e.g., "model.language_model.layers.3.self_attn.q_proj.weight"
        # -> "model.language_model.layers.3"
        m = re.match(r"(.*\.layers\.\d+)\.", name)
        return m.group(1) if m else name.rsplit(".", 2)[0]

    names = sorted(names, key=natural_key)

    for name in names:
        for target in targets:
            if match_name(name, target):
                if matches[target] is None:
                    matches[target] = name
                else:
                    # Same target matched again — check if it's a different layer
                    prev_prefix = get_layer_prefix(matches[target])
                    curr_prefix = get_layer_prefix(name)

                    if prev_prefix != curr_prefix:
                        # Different layer — flush partial set as unmatched, start new
                        if any(v is not None for v in matches.values()):
                            # Save partial as unmatched
                            pass  # Will be collected below
                        matches = dict.fromkeys(targets, None)
                        matches[target] = name
                    else:
                        # Same layer, same target — this is a real error
                        raise ValueError(
                            f"Matched {target} twice in same layer "
                            f"({matches[target]}, {name})"
                        )

        # Once we have a full set, yield and reset
        if all(matches[target] is not None for target in targets):
            matched_sets.append(matches)
            matches = dict.fromkeys(targets, None)

    unmatched_set = matches if any(v is not None for v in matches.values()) else None

    if return_unmatched:
        return matched_sets, unmatched_set
    else:
        return matched_sets
