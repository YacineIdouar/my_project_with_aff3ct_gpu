import argparse
import os
import math
import time

# Must be set before TensorFlow is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide INFO/WARNING, keep errors

import tensorflow as tf

import sionna
from sionna.phy.mapping import BinarySource, Mapper, Demapper, Constellation
from sionna.phy.channel.awgn import AWGN
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.fec.ldpc import LDPC5GDecoder

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser()

parser.add_argument("-K", type=int, required=True)
parser.add_argument("-N", type=int, required=True)

parser.add_argument("-m", type=float, required=True)
parser.add_argument("-M", type=float, required=True)
parser.add_argument("-s", type=float, required=True)

parser.add_argument("--dec-implem", type=str, choices=["BOXPLUS", "SPA", "BOXPLUS-PHI", "MINSUM", "MS", "OFFSET-MINSUM", "OMS"], default="SPA")
parser.add_argument("--dec-offset", type=float, default=0.5)

parser.add_argument("-i", type=int, default=20)
parser.add_argument("-e", type=int, default=100)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--unroll", type=int, default=8)
parser.add_argument("--no-xla", action="store_true",
                    help="disable XLA/JIT and run the graph in plain TF (slower, but a useful "
                         "fallback if a layer is not XLA-compatible)")

args = parser.parse_args()

use_xla = not args.no_xla

# ----------------------------
# Define the device (GPU if available)
# ----------------------------
# TensorFlow places ops automatically, there is no explicit .to(device) as in the PyTorch
# backend. Memory growth has to be requested before the first allocation, otherwise TF
# pre-reserves the whole device and any other process on the GPU starves.
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass  # already initialized, keep the default allocator
device_type = "CUDA" if gpus else "CPU"

# Some Sionna layers switch to XLA-friendly (but slightly slower in eager) implementations
# when this flag is set; it is required for the LDPC decoder / demapper under jit_compile.
if use_xla:
    try:
        sionna.phy.config.xla_compat = True
    except AttributeError:
        pass  # not exposed by this Sionna release

# ----------------------------
# Build communication objects
# ----------------------------

# Instantiate a source
source = BinarySource()

# Instantiate BPSK modulation/mapping
bps = 1 # number of bits per symbol
constellation = Constellation("pam", bps)
mapper = Mapper(constellation=constellation)

# Instantiate LDPC encoder/decoder (5G NR UL-SCH)
encoder = LDPC5GEncoder(k=args.K, n=args.N)

dec_implem = args.dec_implem
if args.dec_implem == "SPA":
    dec_implem = "BOXPLUS"
elif args.dec_implem == "MS":
    dec_implem = "MINSUM"
elif args.dec_implem == "OMS":
    dec_implem = "OFFSET-MINSUM"

dec_kwargs = dict(
    num_iter=args.i,
    hard_out=True,
    cn_update=dec_implem.lower(),
)
try:
    decoder = LDPC5GDecoder(encoder, offset=args.dec_offset, **dec_kwargs)
except TypeError:
    # Older 1.x point releases do not expose 'offset' on the decoder itself.
    decoder = LDPC5GDecoder(encoder, **dec_kwargs)
    if dec_implem == "OFFSET-MINSUM":
        print(f"# (WW) this Sionna release ignores --dec-offset, using its built-in default")

# Instantiate AWGN channel
channel = AWGN()

# Instantiate BPSK demodulation/demapping
demapping_method="app"
demapper = Demapper(demapping_method, constellation=constellation)

# ----------------------------
# Header
# ----------------------------

def print_header(args, encoder):
    print(f"# ==============================================================================")
    print(f"# ===---   MiniFEC - Powered by Sionna {sionna.__version__} and TensorFlow {tf.__version__}   ---===")
    print(f"# ==============================================================================")
    print(f"# Parameters:")
    print(f"# * Simulation ------------------------------------")
    print(f"#    ** SNR Eb/N0 range : {args.m} → {args.M} dB (step {args.s})")
    print(f"#    ** Batch size      : {args.batch}")
    print(f"#    ** Backend         : {device_type}")
    print(f"#    ** Compilation     : {'tf.function + XLA (jit_compile)' if use_xla else 'tf.function (graph only)'}")
    print(f"#    ** Unroll factor   : {args.unroll} batch(es) per host sync")
    print(f"# * Codec -----------------------------------------")
    print(f"#    ** Type            : LDPC")
    print(f"#    ** K (info bits)   : {args.K}")
    print(f"#    ** N (frame length): {args.N}")
    print(f"#    ** Code rate       : {args.K / args.N}")
    print(f"# * Encoder ---------------------------------------")
    print(f"#    ** K_LDPC          : {encoder._k_ldpc}")
    print(f"#    ** N_LDPC          : {encoder._n_ldpc}")
    print(f"#    ** Base Graph      : {encoder._bg} (auto-selected)")
    print(f"# * Decoder ---------------------------------------")
    print(f"#    ** Iterations      : {args.i}")
    print(f"#    ** Scheduling      : BP_FLOODING")
    print(f"#    ** Update rule     : {args.dec_implem.upper()}")
    print(f"#    ** Syndrome        : OFF")
    if (dec_implem.upper() == "OFFSET-MINSUM"):
        print(f"#    ** OMS offset      : {args.dec_offset}")
    print(f"# * Monitor ---------------------------------------")
    print(f"#    ** Frame error cnt : {args.e}")
    print(f"#")
    print(f"# The simulation is running...")

print_header(args, encoder)

# ----------------------------
# Helper metrics
# ----------------------------

def print_line(sigma, esn0, ebn0, fra, be, fe, ber, fer, thr, et):
    print(
        f"   "
        f"{sigma:8.4f} | {esn0:8.2f} | {ebn0:8.2f} || "
        f"{fra:10d} | {be:8d} | {fe:8d} | {ber:8.2e} | {fer:8.2e} || "
        f"{thr:8.3f} | {et:>8}"
    )

# ----------------------------
# Legend
# ----------------------------

def print_legend():
    print("# --------------------------------||--------------------------------------------------------||---------------------")
    print("#        Signal Noise Ratio       ||    Bit Error Rate (BER) and Frame Error Rate (FER)     ||  Global throughput  ")
    print("#              (SNR)              ||                                                        ||  and elapsed time   ")
    print("# --------------------------------||--------------------------------------------------------||---------------------")
    print("# ----------|----------|----------||------------|----------|----------|----------|----------||----------|----------")
    print("#     Sigma |    Es/N0 |    Eb/N0 ||        FRA |       BE |       FE |      BER |      FER ||  SIM_THR |    ET/RT ")
    print("#           |     (dB) |     (dB) ||            |          |          |          |          ||   (Mb/s) | (hhmmss) ")
    print("# ----------|----------|----------||------------|----------|----------|----------|----------||----------|----------")

print_legend()

# ----------------------------
# Compile inner simu step
# ----------------------------

def sim_step(no):
    # 1. source
    u = source([args.batch, args.K])
    # 2. LDPC encoding
    codeword = encoder(u)
    # 3. BPSK modulation
    x = mapper(codeword)
    # 4. channel AWGN
    y = channel(x, no)
    # 5. BPSK demapper
    llr = demapper(y, no)
    # 6. LDPC decoding
    v = decoder(llr)
    # 7. count number of bit errors
    bit_errors = tf.reduce_sum(tf.cast(tf.not_equal(v, u), tf.int64), axis=1)
    return bit_errors

n_traces = 0  # only incremented while tracing, so it counts (re)compilations

# /!\ this step is a key point for high throughput and low latency /!\
#
# Three things matter here, and only the first one has a PyTorch counterpart:
#
#  1. jit_compile=True hands the whole chain to XLA (the same compiler JAX uses): the
#     encoder/mapper/channel/demapper/decoder are fused into a handful of kernels instead
#     of one launch per op, which is what torch.compile(mode="max-autotune") does.
#  2. The unroll loop lives *inside* the graph. In the PyTorch version the Python loop
#     re-enters the compiled region 'unroll' times; here the whole mini-batch of frames is
#     a single graph launch, so the host touches the device exactly once per outer
#     iteration, and XLA gets to keep the decoder state in registers across the loop.
#  3. 'no' is a scalar tf.Tensor, never a Python float. A Python float is baked into the
#     graph as a constant and forces a full retrace + XLA recompile at every SNR point.
@tf.function(jit_compile=use_xla, reduce_retracing=True)
def sim_mini_batch(no):
    global n_traces
    n_traces += 1

    total_bit_errors = tf.zeros((), dtype=tf.int64)
    total_frame_errors = tf.zeros((), dtype=tf.int64)

    # autograph turns this into a tf.while_loop, so the graph stays small (one body) no
    # matter how large --unroll is: big unroll factors cost nothing at compile time.
    for _ in tf.range(args.unroll):
        bit_errors = sim_step(no)
        total_bit_errors += tf.reduce_sum(bit_errors)
        total_frame_errors += tf.reduce_sum(tf.cast(bit_errors > 0, tf.int64))

    return total_bit_errors, total_frame_errors

# ----------------------------
# MAIN LOOP
# ----------------------------

# Warm-up: trace + XLA-compile the graph once, outside of any measured region. Without it
# the first SNR point pays the whole compilation cost and its throughput is meaningless.
warmup_start = time.time()
sim_mini_batch(tf.constant(1.0, tf.float32))
print(f"# Graph traced and compiled in {time.time() - warmup_start:.1f}s, starting the simulation")
print(f"#")

rate = args.K / args.N
n_snr = int(math.floor((args.M - args.m) / args.s + 1e-9)) + 1
SNRs = [args.m + i * args.s for i in range(n_snr)]
for ebn0 in SNRs:

    esn0 = ebn0 + 10 * math.log10(rate * bps)
    sigma = math.sqrt(1.0 / (2 * (10 ** (esn0 / 10))))
    n0 = 2 * (sigma ** 2)

    # built once per SNR point: same dtype and shape as the warm-up call, so the graph
    # above is reused as-is and n_traces stays at 1 for the whole run
    no = tf.constant(n0, tf.float32)

    frame_errors = 0
    total_bits = 0
    total_bit_errors = 0
    total_frames = 0

    start = time.time()

    while frame_errors < args.e:

        # run a mini-batch of iterations on GPU before syncing
        tf_bit_errors, tf_frame_errors = sim_mini_batch(no)

        # sync once per mini-batch of args.unroll iterations (.numpy() blocks until the
        # device is done, which is also what makes the timing below accurate)
        total_bit_errors += int(tf_bit_errors.numpy())
        frame_errors += int(tf_frame_errors.numpy())
        total_bits += args.unroll * args.batch * args.K
        total_frames += args.unroll * args.batch

    end = time.time()
    elapsed = end - start

    ber = total_bit_errors / total_bits
    fer = frame_errors / total_frames

    throughput = (total_bits / elapsed) / 1e6 # Mb/s

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print_line(
        sigma,
        esn0,
        ebn0,
        total_frames,
        total_bit_errors,
        frame_errors,
        ber,
        fer,
        throughput,
        f"{h:02d}h{m:02d}'{s:02d}"
    )

if n_traces > 1:
    print(f"# (WW) the graph was traced {n_traces} times: something is forcing recompilation "
          f"and the throughput above is pessimistic")
print("# End of the simulation.")
