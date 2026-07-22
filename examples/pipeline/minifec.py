import argparse
import time
import torch
import math

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

args = parser.parse_args()

# ----------------------------
# Define the device (GPU if available)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

decoder = LDPC5GDecoder(
    encoder,
    num_iter=args.i,
    hard_out=True,
    cn_update=dec_implem.lower(),
    offset=args.dec_offset
)

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
    print(f"# ===---       MiniFEC - Powered by Sionna 2.0.1 and PyTorch 2.12.1       ---===")
    print(f"# ==============================================================================")
    print(f"# Parameters:")
    print(f"# * Simulation ------------------------------------")
    print(f"#    ** SNR Eb/N0 range : {args.m} → {args.M} dB (step {args.s})")
    print(f"#    ** Batch size      : {args.batch}")
    print(f"#    ** Backend         : {device.type.upper()}")
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

def sim_step(n0, source, encoder, mapper, channel, demapper, decoder):
    # 1. source
    u = source([args.batch, args.K])
    # 2. LDPC encoding
    codeword = encoder(u)
    # 3. BPSK modulation
    x = mapper(codeword)
    # 4. channel AWGN
    y = channel(x, n0)
    # 5. BPSK demapper
    llr = demapper(y, n0)
    # 6. LDPC decoding
    v = decoder(llr)
    # 7. count number of bit errors
    bit_errors = (v != u).sum(dim=1)
    return bit_errors

# /!\ this step is a key point for high throughput and low latency /!\
sim_step_compiled = torch.compile(sim_step, mode="max-autotune")

# ----------------------------
# MAIN LOOP
# ----------------------------

# set compute tasks to device
source = source.to(device)
encoder = encoder.to(device)
decoder = decoder.to(device)
mapper = mapper.to(device)
demapper = demapper.to(device)

rate = args.K / args.N
SNRs = torch.arange(args.m, args.M + 1e-9, args.s)
for ebn0 in SNRs:

    esn0 = ebn0 + 10 * torch.log10(torch.tensor(rate) * bps).item()
    sigma = math.sqrt(1.0 / (2 * (10 ** (esn0 / 10))))
    n0 = 2 * (sigma ** 2)

    frame_errors = 0
    total_bits = 0
    total_bit_errors = 0
    total_frames = 0

    start = time.time()
    with torch.inference_mode():

        while frame_errors < args.e:

            # run a mini-batch of iterations on GPU before syncing
            torch_frame_errors = torch.zeros((), device=device, dtype=torch.int64)
            torch_total_bit_errors = torch.zeros((), device=device, dtype=torch.int64)

            unroll = 8
            for _ in range(unroll):
                bit_errors = sim_step_compiled(n0, source, encoder, mapper, channel, demapper, decoder)
                torch_total_bit_errors += bit_errors.sum()
                torch_frame_errors += (bit_errors > 0).sum()

            # sync once per mini-batch of 8 iterations
            frame_errors += torch_frame_errors.item()
            total_bit_errors += torch_total_bit_errors.item()
            total_bits += unroll * args.batch * args.K
            total_frames += unroll * args.batch

    if device.type == "cuda":
        torch.cuda.synchronize()

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

print("# End of the simulation.")
