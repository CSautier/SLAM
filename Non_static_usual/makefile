CC=clang++
IDIR =./include
CFLAGS= -I$(IDIR) -std=c++11
ODIR=obj
LIBS=-lboost_system	-lboost_filesystem	-ltbb -ltbbmalloc -lgtsam
DEPS = objects.h
OBJ = main.o

_DEPS = objects.h	maxmixture_factor.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)


.PHONY: clean


clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ main ./*.p temp Slam.gif
