CC=clang++
CFLAGS1=
CFLAGS=-I /usr/include/opencv2 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_calib3d -lopencv_video	-lopencv_features2d	-lboost_system	-lboost_filesystem	-ltbb	-lgtsam 
DEPS = optimizer.hpp
OBJ = main.o optimizer.o 

%.o: %.cpp $(DEPS)

	$(CC) -c -o $@ $< $(CFLAGS1)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)



.PHONY: clean

clean:
	rm *.o
