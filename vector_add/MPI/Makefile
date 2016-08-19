CXX = mpicxx
CXXFLAGS = -std=c++11 -g -O2

.default: parallel_vector_add

parallel_vector_add:
	$(CXX) $(CXXFLAGS) -o parallel_vector_add parallel_vector_add.cpp

clean:
	rm parallel_vector_add
