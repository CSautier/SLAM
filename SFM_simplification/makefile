CC=clang++
CFLAGS1=
CFLAGS=-I /usr/include/opencv2 `pkg-config --cflags --libs opencv`	-lboost_system	-lboost_filesystem	-ltbb	-lgtsam -ltbbmalloc
DEPS = optimizer.hpp
OBJ = main.o optimizer.o 

%.o: %.cpp $(DEPS)

	$(CC) -O3 -c -o $@ $< $(CFLAGS1)

main: $(OBJ)
	$(CC) -O3 -o $@ $^ $(CFLAGS)



.PHONY: clean

clean:
	rm *.o
