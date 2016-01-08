OUT := dirspace
SRC := $(wildcard *.c)
OBJS := ${SRC:.c=.o}

CFLAGS := 
LDFLAGS := 

ifeq ($(DEBUG),1)
  CFLAGS += -g -O0
else
  CFLAGS += -O3 -DNDEBUG
endif

CFLAGS += -std=c11

-include localvars.mk

all: $(OUT)

%.o: %.c helptext
	$(CC) -c $< $(CFLAGS)

$(OUT): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

helptext: info.txt addhelp.py
	./addhelp.py

.PHONY: clean
clean:
	rm -f $(OUT) $(OBJS)

-include localrules.mk
