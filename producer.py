import yfinance as yf
import pandas as pd
import struct
import posix_ipc
import mmap
import signal
import time

SHM_NAME = "/stock_data"  # Boost interprocess strips leading slashes internally on macOS mapping
BUFFER_CAPACITY = 100
HEADER_FMT = '<qqq'  # Write index, Read index, Capacity (24 bytes)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Fix your packet size to match C++ structural requirements strictly: 
# 4 bytes ticker + 25 bytes datetime + 20 bytes floats = 49 bytes
DATA_PACKET_FMT = '<4s25sfffff'
DATA_PACKET_SIZE = struct.calcsize(DATA_PACKET_FMT)
TOTAL_SHM_SIZE = HEADER_SIZE + (DATA_PACKET_SIZE * BUFFER_CAPACITY)

def handle_interrupt(signum, frame):
    print("\nInterrupt received. Cleaning up shared memory.")
    try:
        posix_ipc.unlink_shared_memory(SHM_NAME)
    except Exception:
        pass
    exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

def fetch_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d", interval="1m")[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.reset_index(inplace=True)
    df["Datetime"] = df["Datetime"].astype(str)
    return df

ticker = 'AAPL'
df = fetch_data(ticker)

try:
    posix_ipc.unlink_shared_memory(SHM_NAME)
except Exception:
    pass

memory = posix_ipc.SharedMemory(SHM_NAME, posix_ipc.O_CREX, size=TOTAL_SHM_SIZE)
mapfile = mmap.mmap(memory.fd, memory.size)
memory.close_fd()

# Initial Header write (write_idx=0, read_idx=0, capacity=100)
mapfile.seek(0)
mapfile.write(struct.pack(HEADER_FMT, 0, 0, BUFFER_CAPACITY))
print(f"\nRing buffer initialized with total {TOTAL_SHM_SIZE} BYTES\n")

def write_to_ring(ticker: str, df: pd.DataFrame, idx) -> bool:
    mapfile.seek(0)
    write_idx, read_idx, capacity = struct.unpack(HEADER_FMT, mapfile.read(HEADER_SIZE))

    # SPSC Full Buffer check
    if (write_idx + 1) % capacity == read_idx:
        return False
    
    # Calculate offset
    slot_offset = HEADER_SIZE + (write_idx * DATA_PACKET_SIZE)
    row = df.iloc[idx]
    
    # Ensure bytes fit exact sizes using ljust or clipping
    ticker_bytes = ticker.encode('utf-8')[:4].ljust(4, b'\x00')
    dt_bytes = str(row["Datetime"]).encode('utf-8')[:25].ljust(25, b'\x00')

    packed_data = struct.pack(
        DATA_PACKET_FMT,
        ticker_bytes,
        dt_bytes,
        float(row["Open"]),
        float(row["High"]),
        float(row["Low"]),
        float(row["Close"]),
        float(row["Volume"]),
    )
    
    mapfile.seek(slot_offset)
    mapfile.write(packed_data)

    # FIX: Update ONLY the write index without destroying the rest of the header layout
    new_write_idx = (write_idx + 1) % capacity
    mapfile.seek(0)
    mapfile.write(struct.pack('<q', new_write_idx))
    return True

def stream_data(ticker: str, df: pd.DataFrame):
    for i in range(len(df)):
        while not write_to_ring(ticker, df, i):
            time.sleep(0.0001) # Avoid hot pinning python thread
        print(f"Python Packed slot {i}")
        time.sleep(0.1) # Simulate real-time delay feed
    
    print("\nFinished writing data. Keeping SHM segment open for consumer processing...")
    # Keep segment alive until manual termination
    while True:
        time.sleep(1)

stream_data(ticker=ticker, df=df)
