SOURCE:=mm
TARGET:=$(SOURCE).out

.PHONY: $(TARGET) clean run

all: $(TARGET)

$(TARGET):
	nvcc -o $(TARGET) $(SOURCE).cu -lcudart

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
