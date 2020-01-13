GHC_FLAGS = -O3 -Wall -rtsopts -XBangPatterns -eventlog -threaded

all: collatz

collatz: collatz.hs
	ghc --make ${GHC_FLAGS} collatz.hs -o $@
	ghc ${GHC_FLAGS} collatz.hs -c -ddump-simpl > collatz.hs.core
	ghc --show-iface collatz.hi > collatz.hi.txt

clean:
	rm -f collatz collatz.o collatz.hs.core collatz.hi collatz.hi.txt
