CXX = g++
BOOST_PATH = /opt/homebrew/include
CXXFLAGS = -std=c++17 -O2
SRC = consumer.cpp
TARGET = consumer

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
