OUTPUT_FOLDER = bin

all: serial parallel

parallel:
# TODO : Parallel compilation

serial:
	g++ src/serial/serial.cpp -o $(OUTPUT_FOLDER)/serial