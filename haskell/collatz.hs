import Data.Bits
import System.Random

collatz :: Int -> Integer -> Int
collatz !len 1 = len
collatz !len !x =
    if testBit x 0
    then collatz (2 + len) (((x `shiftL` 1) + x + 1) `shiftR` 1)
    else collatz (1 + len) (x `shiftR` 1)

longer :: (Integer, Int) -> [Integer] -> [(Integer, Int)]
longer _ [] = []
longer (a, b) (r:rs) =
    let !x = (r, (collatz 0 r))
    in  if (snd x) > b
        then
            x:(longer x rs)
        else
            longer (a, b) rs

main :: IO ()
main = do
    g <- getStdGen
    let rs = filter odd $ randomRs (2::Integer, (10::Integer)^(100::Integer)) g
    mapM_ print $ longer (1, 0) rs
{-
    print $ collatz (1, 0) 0x00000e7f9b9cce9b9b9cce9b9b9cce9b9b9cce9b9b9cce9b9b9cce9b9b9cce9b9b9cce9b9b9cce9b9b9cce9b
-}
