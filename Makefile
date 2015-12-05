OUT := dirspace
SRC := $(wildcard *.c)
OBJS := ${SRC:.c=.o}

CFLAGS := 
LDFLAGS := 

ifeq ($(FAST),1)
  CFLAGS += -O3 -DNDEBUG
else
  CFLAGS += -g -O0
endif

CFLAGS += -Werror -Wall

all: $(OUT)

%.o: %.c
	$(CC) -c $< $(CFLAGS)

$(OUT): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: test
test: $(OUT)
	./$(OUT) ..

.PHONY: test
retest: clean test

.PHONY: clean
clean:
	rm -f $(OUT) $(OBJS)
