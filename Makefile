Cpp := g++
CppFLAGS := -std=c++17 -Wall -Wextra -g

BINDIR := bin
OBJDIR := obj
SRCDIR := src
TARGET := $(BINDIR)/main

SRCS := main.cpp NN.cpp Layer.cpp Matrix.cpp Activation.cpp
OBJS := $(SRCS:%.cpp=$(OBJDIR)/%.o)

.PHONY: all exe clean run

all: exe

exe: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	$(Cpp) $(CppFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(Cpp) $(CppFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

run: exe
	./$(TARGET)

clean:
	rm -rf $(OBJDIR) $(BINDIR)