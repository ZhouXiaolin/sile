"""Reference results for the cutile examples, computed with numpy.

Usage: numpy_ref.py OP NIN LEN... SCALAR... < inputs.f32le > expected.f32le

NIN is the number of input arrays; each LEN is one array's element count,
read from stdin in order. The scalars follow as text and each op parses its
own, which keeps numeric scalars out of the length list.
"""

import sys

import numpy as np


def read_inputs(count, lengths):
    data = sys.stdin.buffer.read()
    arrays = []
    offset = 0
    for length in lengths[:count]:
        n = int(length)
        arrays.append(np.frombuffer(data, dtype="<f4", count=n, offset=offset))
        offset += n * 4
    return arrays


def softmax_rows(scores):
    peak = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - peak)
    return weights / weights.sum(axis=-1, keepdims=True)


def flash_attention(q, k, v, m, d, qk_scale, input_pos, causal):
    # One head plane at a time. The planes fold the batch in, so a GQA group
    # is q planes over kv planes, and a causal mask keeps the keys up to the
    # query position plus input_pos.
    q_heads = q.reshape(-1, m, d)
    k_heads = k.reshape(-1, m, d)
    v_heads = v.reshape(-1, m, d)
    group = len(q_heads) // len(k_heads)
    rows = np.arange(m)
    allowed = rows[:, None] + input_pos
    out = np.zeros_like(q_heads)
    for head in range(len(q_heads)):
        kv = head // group
        scores = (q_heads[head] @ k_heads[kv].T) * np.float32(qk_scale)
        if causal:
            scores = np.where(rows[None, :] <= allowed, scores, -np.inf)
        out[head] = softmax_rows(scores) @ v_heads[kv]
    return out.reshape(-1)


def main():
    op = sys.argv[1]
    count = int(sys.argv[2])
    lengths = sys.argv[3:]
    scalars = lengths[count:]
    lengths = lengths[:count]

    if op == "saxpy":
        x, y = read_inputs(count, lengths)
        alpha = np.float32(float(scalars[0]))
        expected = alpha * x + y
    elif op == "dropout":
        x, keep = read_inputs(count, lengths)
        p = np.float32(float(scalars[0]))
        kept = keep > p
        expected = np.where(kept, x / (np.float32(1.0) - p), np.float32(0.0))
    elif op == "matmul":
        a, b = read_inputs(count, lengths)
        m, k, n = (int(s) for s in scalars)
        expected = (
            a.reshape(m, k).astype(np.float64) @ b.reshape(k, n).astype(np.float64)
        )
    elif op == "batch_matmul":
        a, b = read_inputs(count, lengths)
        batch, m, k, n = (int(s) for s in scalars)
        left = a.reshape(batch, m, k).astype(np.float64)
        right = b.reshape(batch, k, n).astype(np.float64)
        expected = (left @ right).reshape(-1)
    elif op == "softmax":
        x, = read_inputs(count, lengths)
        m, n = (int(s) for s in scalars)
        expected = softmax_rows(x.reshape(m, n).astype(np.float64))
    elif op == "rms_norm":
        x, w = read_inputs(count, lengths)
        m, n = (int(s) for s in scalars[:2])
        eps = float(scalars[2])
        rows = x.reshape(m, n).astype(np.float64)
        scale = 1.0 / np.sqrt((rows * rows).mean(axis=1, keepdims=True) + eps)
        expected = rows * scale * w.reshape(1, n).astype(np.float64)
    elif op == "flash_attention":
        q, k, v = read_inputs(count, lengths)
        m, d = (int(s) for s in scalars[:2])
        qk_scale = float(scalars[2])
        expected = flash_attention(q, k, v, m, d, qk_scale, 0, False)
    elif op == "flash_attention_causal":
        q, k, v = read_inputs(count, lengths)
        m, d = (int(s) for s in scalars[:2])
        qk_scale = float(scalars[2])
        input_pos = int(scalars[3])
        expected = flash_attention(q, k, v, m, d, qk_scale, input_pos, True)
    else:
        raise SystemExit(f"unknown op {op}")

    sys.stdout.buffer.write(expected.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
