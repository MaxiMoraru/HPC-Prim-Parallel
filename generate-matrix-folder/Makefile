CC = gcc
CFLAGS = -Wall -Wextra -O2

all: generate-matrix-c

generate-matrix-c: generate-matrix.c
	$(CC) $(CFLAGS) -o $@ $<

test: generate-matrix
	./generate-matrix

clean:
	rm -f generate-matrix
