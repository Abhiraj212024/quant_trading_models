#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <string>

const size_t TICKER_SIZE = 4;
const size_t DATETIME_SIZE = 25;
const char* SHM_NAME = "/stock_data";
const size_t BUFFER_CAPACITY = 100;
const size_t HEADER_SIZE = 24;

#pragma pack(push, 1)
struct TickData {
    char ticker[TICKER_SIZE];
    char datetime[DATETIME_SIZE];
    float open;
    float high;
    float low;
    float close;
    float volume;
};
#pragma pack(pop)

const size_t TOTAL_SHM_SIZE = HEADER_SIZE + (BUFFER_CAPACITY * sizeof(TickData));

void run_engine() {
    while (true) { // Outer loop to allow restarting sessions
        std::cout << "\n[C++] Waiting for Python to initialize shared memory...\n";
        
        int shm_fd = -1;
        while (shm_fd < 0) {
            shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
            if (shm_fd < 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }

        void* shm_base = mmap(NULL, TOTAL_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shm_base == MAP_FAILED) {
            std::cerr << "[C++] Mapping memory failed!\n";
            close(shm_fd);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        
        std::atomic<uint64_t>* write_idx = reinterpret_cast<std::atomic<uint64_t>*>(shm_base);
        std::atomic<uint64_t>* read_idx  = reinterpret_cast<std::atomic<uint64_t>*>(static_cast<char*>(shm_base) + 8);
        std::atomic<uint64_t>* capacity  = reinterpret_cast<std::atomic<uint64_t>*>(static_cast<char*>(shm_base) + 16);
        char* buffer_start = static_cast<char*>(shm_base) + HEADER_SIZE;

        std::cout << "[C++] Connected! Interpreting Ticks...\n";

        while (true) {
            // Check for Poison Pillar: If capacity is wiped to 0 by Python, disconnect
            if (capacity->load(std::memory_order_relaxed) == 0) {
                std::cout << "[C++] Python disconnected or terminated. Resetting session...\n";
                break;
            }

            uint64_t curr_write = write_idx->load(std::memory_order_acquire);
            uint64_t curr_read  = read_idx->load(std::memory_order_relaxed);

            if (curr_read != curr_write) {
                TickData* tick = reinterpret_cast<TickData*>(buffer_start + (curr_read * sizeof(TickData)));

                std::string ticker_str(tick->ticker, TICKER_SIZE);
                std::string datetime_str(tick->datetime, DATETIME_SIZE);

                std::cout << "C++ Consumed -> " << ticker_str << " | " 
                          << "Datetime: "<< datetime_str << " | " 
                          << "Open: " << tick->open << " | "
                          << "High: " << tick->high << " | "
                          << "Low: "  << tick->low << " | "
                          << "Volume: " << tick->volume << "\n";

                read_idx->store((curr_read + 1) % BUFFER_CAPACITY, std::memory_order_release);
            } else {
                std::this_thread::yield();
            }
        }

        // Clean up handles completely so macOS can garbage collect this segment
        munmap(shm_base, TOTAL_SHM_SIZE);
        close(shm_fd);
    }
}

int main() {
    run_engine();
    return 0;
}
